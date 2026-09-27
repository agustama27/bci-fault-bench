"""Pipeline bajo prueba + recolector de telemetría.

Recibe el flujo EEG y el de marcadores por LSL, segmenta cada ensayo
[onset+2, onset+6] s, filtra 8-30 Hz, decodifica con CSP+LDA congelado y
registra por ensayo y por ventana de 1 s la telemetría del flujo y del proceso.

Salidas en --outdir:
  trials.csv     una fila por ensayo izquierda/derecha
  telemetry.csv  una fila por segundo nominal desde t0
  consumer.json  resumen
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from collections import deque

import numpy as np
from mne.filter import filter_data
from mne_lsl.lsl import StreamInlet, local_clock, resolve_streams

from . import data
from .decoder import Decoder

RESOLVE_TIMEOUT = 90.0
TRIAL_TIMEOUT_S = 3.0          # espera máxima por los datos de un ensayo tras su fin nominal
PAD_S = 1.0                    # relleno para filtrar sin efecto de borde
IDLE_ABORT_S = 30.0            # sin datos tras el inicio → se da por terminada
DECODED = (1, 2)               # left_hand, right_hand


def _resolve(name: str):
    t_end = local_clock() + RESOLVE_TIMEOUT
    while local_clock() < t_end:
        s = resolve_streams(timeout=1.0, name=name)
        if s:
            return s[0]
    raise RuntimeError(f"no se encontró el flujo {name}")


class Telemetry:
    """Acumula chunks y produce una fila por segundo nominal."""
    def __init__(self, sfreq):
        self.sfreq = sfreq
        self.rows = {}
        self.last_ts = None
        self.last_arrival = None
        self.exceptions = 0
        self.stalls = 0

    def _row(self, sec: int):
        if sec not in self.rows:
            self.rows[sec] = dict(sec=sec, n_samples=0, n_chunks=0, n_gaps=0, gap_s=0.0,
                                  lat_sum=0.0, lat_max=0.0, ia=[], preds=0, exceptions=0)
        return self.rows[sec]

    def on_chunk(self, ts: np.ndarray, arrival: float, t0: float):
        r = self._row(int(ts[-1] - t0))
        r["n_samples"] += len(ts)
        r["n_chunks"] += 1
        lat = arrival - ts[-1]
        r["lat_sum"] += lat
        r["lat_max"] = max(r["lat_max"], lat)
        if self.last_arrival is not None:
            r["ia"].append(arrival - self.last_arrival)
        self.last_arrival = arrival
        seq = np.concatenate([[self.last_ts], ts]) if self.last_ts is not None else ts
        d = np.diff(seq)
        gaps = d > 1.5 / self.sfreq
        if gaps.any():
            r["n_gaps"] += int(gaps.sum())
            r["gap_s"] += float((d[gaps] - 1.0 / self.sfreq).sum())
        self.last_ts = float(ts[-1])

    def on_pred(self, t_nom: float, t0: float):
        self._row(int(t_nom - t0))["preds"] += 1

    def on_exception(self, t_now: float, t0: float):
        self.exceptions += 1
        self._row(int(max(0.0, t_now - t0)))["exceptions"] += 1

    def dump(self, path: str, t0: float, duration: float):
        n_expected = int(round(self.sfreq))
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sec", "n_expected", "n_samples", "n_chunks", "n_gaps", "gap_s",
                        "lat_mean", "lat_max", "ia_mean", "ia_std", "ia_max", "preds", "exceptions"])
            for sec in range(int(np.ceil(duration))):
                r = self.rows.get(sec)
                if r is None:
                    w.writerow([sec, n_expected, 0, 0, 0, 1.0, "", "", "", "", "", 0, 0])
                    continue
                ia = np.array(r["ia"]) if r["ia"] else np.array([np.nan])
                w.writerow([sec, n_expected, r["n_samples"], r["n_chunks"], r["n_gaps"],
                            round(r["gap_s"], 4),
                            round(r["lat_sum"] / max(r["n_chunks"], 1), 5), round(r["lat_max"], 5),
                            round(float(np.nanmean(ia)), 5), round(float(np.nanstd(ia)), 5),
                            round(float(np.nanmax(ia)), 5), r["preds"], r["exceptions"]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--duration", type=float, default=900.0,
                    help="duración máxima de respaldo (s); el fin real lo marca el reproductor")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    dec = Decoder.load(a.model)

    einfo = _resolve(f"{a.exec_id}-eeg")
    minfo = _resolve(f"{a.exec_id}-markers")
    eeg = StreamInlet(einfo, chunk_size=0, max_buffered=360, recover=True)
    mk = StreamInlet(minfo, chunk_size=0, max_buffered=360, recover=True)
    eeg.open_stream()
    mk.open_stream()
    sfreq, n_ch = float(einfo.sfreq), int(einfo.n_channels)
    tel = Telemetry(sfreq)

    buf_ts, buf_x = deque(), deque()     # bloques recibidos (ts array, x array)
    buf_len_s = 30.0
    pending = []                         # (onset_ts, code)
    trials = []
    t0 = None
    ended_at = None                      # marca de fin nominal enviada por el reproductor
    started_wall = local_clock()
    last_data_wall = local_clock()
    exc_total = 0

    def _segment(t_a: float, t_b: float):
        ts = np.concatenate(buf_ts) if buf_ts else np.empty(0)
        if len(ts) == 0:
            return None, None
        x = np.concatenate(buf_x)
        m = (ts >= t_a) & (ts < t_b)
        return ts[m], x[m]

    def _decode(onset: float):
        ts, x = _segment(onset + data.TMIN - PAD_S, onset + data.TMAX + PAD_S)
        if ts is None or len(ts) < int(0.5 * sfreq):
            return None
        # El flujo viaja en µV; el decodificador se entrenó en V (unidades de MNE).
        xf = filter_data(x.T.astype(np.float64) * 1e-6, sfreq, data.FMIN, data.FMAX,
                         method="iir", verbose=False)
        m = (ts >= onset + data.TMIN) & (ts < onset + data.TMAX)
        if m.sum() < int(0.5 * sfreq):
            return None
        seg = xf[:, m]
        proba = dec.predict_proba(seg[None])[0]
        pred = int(dec.model.classes_[int(np.argmax(proba))])
        return pred, float(proba.max()), int(m.sum())

    while True:
        now = local_clock()
        try:
            ms, mts = mk.pull_chunk(timeout=0.0, max_samples=64)
            for s, ts in zip(np.asarray(ms).reshape(-1), mts):
                code = int(s)
                if code == 100:
                    t0 = float(ts)
                elif code == 200:
                    ended_at = float(ts)
                elif code in data.EVENT_ID.values():
                    pending.append((float(ts), code))
            xs, ts = eeg.pull_chunk(timeout=0.0, max_samples=4096)
            if len(ts):
                arrival = local_clock()
                last_data_wall = arrival
                if t0 is not None:
                    tel.on_chunk(np.asarray(ts, dtype=float), arrival, t0)
                # pull_chunk devuelve VISTAS sobre un búfer interno que se
                # reutiliza en la próxima llamada: hay que copiar.
                buf_ts.append(np.array(ts, dtype=float, copy=True))
                buf_x.append(np.array(xs, dtype=np.float32, copy=True).reshape(len(ts), n_ch))
                while buf_ts and buf_ts[0][-1] < ts[-1] - buf_len_s:
                    buf_ts.popleft(); buf_x.popleft()
        except Exception:
            exc_total += 1
            if t0 is not None:
                tel.on_exception(now, t0)
            traceback.print_exc()
            time.sleep(0.05)

        # ensayos cuya ventana ya debería haber llegado
        still = []
        for onset, code in pending:
            t_end = onset + data.TMAX + PAD_S
            have = tel.last_ts is not None and tel.last_ts >= t_end
            timed_out = now > t_end + TRIAL_TIMEOUT_S
            if not (have or timed_out):
                still.append((onset, code)); continue
            if code not in DECODED:
                continue
            res = _decode(onset) if have or timed_out else None
            t_dec = local_clock()
            if os.environ.get("BCIBENCH_DEBUG"):
                allts = np.concatenate(buf_ts) if buf_ts else np.empty(0)
                print(f"DBG trial onset_rel={onset - (t0 or 0):.3f} have={have} timed_out={timed_out} "
                      f"buf_n={len(allts)} buf_rel=[{(allts[0] - (t0 or 0)) if len(allts) else 'nan'}, "
                      f"{(allts[-1] - (t0 or 0)) if len(allts) else 'nan'}] res={res}", flush=True)
            if res is None:
                trials.append(dict(onset=onset - (t0 or 0), label=code, pred="", proba="",
                                   n_samples=0, valid=0, decision_delay=t_dec - (onset + data.TMAX)))
            else:
                pred, proba, n = res
                trials.append(dict(onset=onset - (t0 or 0), label=code, pred=pred, proba=round(proba, 4),
                                   n_samples=n, valid=1, decision_delay=round(t_dec - (onset + data.TMAX), 4)))
                if t0 is not None:
                    tel.on_pred(onset + data.TMAX, t0)
        pending = still

        # condiciones de fin: marcador de fin recibido y sin ensayos pendientes,
        # o (respaldo) duración máxima superada
        if ended_at is not None and not pending and now > ended_at + PAD_S + 1.0:
            break
        if t0 is not None and now > t0 + a.duration + TRIAL_TIMEOUT_S + PAD_S + 2.0 and not pending:
            break
        if now - last_data_wall > IDLE_ABORT_S and (t0 is None or now > t0 + 5):
            break
        if t0 is None and now - started_wall > RESOLVE_TIMEOUT:
            break
        time.sleep(0.005)

    # salidas
    with open(os.path.join(a.outdir, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["onset", "label", "pred", "proba", "n_samples", "valid", "decision_delay"])
        w.writeheader(); w.writerows(trials)
    # duración efectiva: la marca de fin del reproductor; si no llegó, la última muestra
    if t0 is not None and ended_at is not None:
        dur_eff = ended_at - t0
    elif t0 is not None and tel.last_ts is not None:
        dur_eff = tel.last_ts - t0
    else:
        dur_eff = a.duration
    tel.dump(os.path.join(a.outdir, "telemetry.csv"), t0 or started_wall, dur_eff)
    valid = [t for t in trials if t["valid"]]
    y = np.array([t["label"] for t in valid]); p = np.array([t["pred"] for t in valid])
    bacc = None
    if len(valid):
        from sklearn.metrics import balanced_accuracy_score
        bacc = float(balanced_accuracy_score(y, p))
    summary = dict(exec_id=a.exec_id, t0=t0, duration_eff=dur_eff, n_trials=len(trials), n_valid=len(valid),
                   balanced_accuracy=bacc, exceptions=exc_total,
                   n_samples_total=int(sum(len(t) for t in buf_ts)) if buf_ts else 0)
    with open(os.path.join(a.outdir, "consumer.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
