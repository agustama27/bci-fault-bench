"""Consumidor alternativo construido sobre la capa de adquisición de BciPy.

Mismo contrato que consumer.py (mismos flujos, mismas ventanas, mismo filtro,
mismo decodificador y los mismos cuatro archivos de salida), pero la recepción
y el búfer los pone el FRAMEWORK: `LslAcquisitionClient` (inlet pylsl creado por
BciPy con max_buflen=365 s, max_chunklen=1, recover=True), su `RingBuffer` y la
consulta por rango de marcas de tiempo `get_data(start, end)`. Lo que se mide es,
entonces, el comportamiento del búfer de BciPy bajo los mismos fallos.

Adaptaciones necesarias (documentadas, no cambian lo medido):
- Resolución POR NOMBRE con tope de 90 s. `resolve_device_stream` de BciPy
  resuelve por tipo ("EEG") sin tope; con 12 ejecuciones en paralelo tomaría el
  flujo de otra. Se reemplaza solo esa función (en el módulo lsl_client) por una
  que devuelve el StreamInfo ya resuelto por nombre; la creación del inlet,
  open_stream, check_device y el RingBuffer siguen siendo los de BciPy.
- BciPy no tiene hilo de recepción: el bucle principal llama a
  `get_latest_data()` cada 5 ms (como consumer.py sondea su inlet). La
  telemetría se engancha sobrescribiendo `get_latest_data` (llama a la de BciPy
  y registra las muestras nuevas del búfer), así también cuentan las muestras
  que trae `get_data`, que internamente vuelve a llamar a `get_latest_data`.
- `start_acquisition` de BciPy consume UNA muestra con `pull_sample` para fijar
  `first_sample_time` y no la guarda en el búfer. En EEG es la primera muestra
  (t0): la telemetría del segundo 0 muestra 249 en vez de 250. En marcadores
  (tasa irregular, timeout 0) podría perderse un marcador ya emitido; por eso el
  cliente de marcadores se inicia ANTES que el de EEG: el reproductor no emite el
  marcador de inicio hasta tener consumidor en ambos flujos.
- `get_data` incluye ambos extremos [start, end]; consumer.py usa [start, end).
  Se descartan las muestras con ts == end para que la ventana sea idéntica.
- Si al vencer el plazo (fin + 3 s) el búfer no llega a `end`, BciPy lanza
  AssertionError ("End time out of range") en lugar de devolver lo parcial: el
  ensayo queda inválido, se cuenta como excepción en la telemetría y su segmento
  se guarda vacío (el framework no entregó datos). consumer.py, en cambio,
  decodifica lo parcial si hay >= 125 muestras. Esa diferencia ES lo observado.

Largo del búfer: MAX_BUFFER_S = 30 s (7500 muestras), igual que el búfer de
consumer.py. Una decisión necesita [onset+1, onset+7] y puede tomarse hasta
3 s después de onset+7 → basta con >= 10 s; 30 s deja margen para ensayos
pendientes superpuestos y para cortes largos. Ojo: el RingBuffer de BciPy se
dimensiona en MUESTRAS, no en segundos: durante un corte no se vacía.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from bisect import bisect_right
from importlib.metadata import version

import numpy as np
from mne.filter import filter_data
from pylsl import StreamInlet, local_clock, resolve_byprop

from bcipy.acquisition import ClientManager, LslAcquisitionClient
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.multimodal import ContentType
from bcipy.acquisition.protocols.lsl import lsl_client as _bcipy_lsl_client
from bcipy.acquisition.protocols.lsl.connect import device_from_metadata

from . import data
from .decoder import Decoder
from .telemetry import Telemetry

RESOLVE_TIMEOUT = 90.0
TRIAL_TIMEOUT_S = 3.0
PAD_S = 1.0
IDLE_ABORT_S = 30.0
DECODED = (1, 2)
MAX_BUFFER_S = 30.0            # segundos de EEG retenidos por el RingBuffer de BciPy
MARKER_BUFFER_N = 1024         # marcadores retenidos (tasa irregular: se cuenta en muestras)
POLL_S = 0.005

_RESOLVED: dict[str, object] = {}


def _resolve(name: str):
    t_end = local_clock() + RESOLVE_TIMEOUT
    while local_clock() < t_end:
        s = resolve_byprop("name", name, timeout=1.0)
        if s:
            return s[0]
    raise RuntimeError(f"no se encontró el flujo {name}")


def _resolve_by_name(device_spec=None):
    """Reemplazo de bcipy...connect.resolve_device_stream: por nombre, ya resuelto."""
    return _RESOLVED[device_spec.name]


_bcipy_lsl_client.resolve_device_stream = _resolve_by_name


def _device_spec(info) -> DeviceSpec:
    """DeviceSpec a partir de los metadatos completos (patrón de discover_device_spec)."""
    tmp = StreamInlet(info)
    full = tmp.info(timeout=10.0)
    base = device_from_metadata(full)
    tmp.close_stream()
    return DeviceSpec(name=base.name, channels=base.channels, sample_rate=base.sample_rate,
                      content_type=base.content_type,
                      data_type="int32" if base.sample_rate == 0 else "float32", static_offset=0.0)


class TelemetryClient(LslAcquisitionClient):
    """LslAcquisitionClient de BciPy + gancho de telemetría sobre get_latest_data."""

    def __init__(self, *args, on_new=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_new = on_new
        self.last_ts = None

    def get_latest_data(self):
        records = super().get_latest_data()
        if records:
            arrival = local_clock()
            i = 0 if self.last_ts is None else bisect_right(records, self.last_ts, key=lambda r: r.timestamp)
            if i < len(records):
                new = records[i:]
                self.last_ts = float(new[-1].timestamp)
                if self.on_new is not None:
                    self.on_new(new, arrival)
        return records


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--duration", type=float, default=900.0)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    dec = Decoder.load(a.model)

    ename, mname = f"{a.exec_id}-eeg", f"{a.exec_id}-markers"
    _RESOLVED[ename] = _resolve(ename)
    _RESOLVED[mname] = _resolve(mname)
    espec, mspec = _device_spec(_RESOLVED[ename]), _device_spec(_RESOLVED[mname])
    sfreq, n_ch = float(espec.sample_rate), int(espec.channel_count)
    tel = Telemetry(sfreq)

    state = dict(t0=None, ended_at=None, last_data_wall=local_clock())
    pending = []
    new_markers = []

    def on_eeg(new, arrival):
        state["last_data_wall"] = arrival
        if state["t0"] is not None:
            tel.on_chunk(np.fromiter((r.timestamp for r in new), dtype=float, count=len(new)),
                         arrival, state["t0"])

    def on_markers(new, arrival):
        new_markers.extend(new)

    eeg = TelemetryClient(max_buffer_len=MAX_BUFFER_S, device_spec=espec, on_new=on_eeg)
    mk = TelemetryClient(max_buffer_len=MARKER_BUFFER_N, device_spec=mspec, on_new=on_markers)
    manager = ClientManager()
    manager.add_client(mk)      # orden de inicio: marcadores primero (ver docstring)
    manager.add_client(eeg)
    manager.start_acquisition()
    eeg = manager.get_client(ContentType.EEG)
    mk = manager.get_client(ContentType.MARKERS)

    trials, segments = [], {}
    started_wall = local_clock()
    exc_total = 0
    n_assert = 0

    def _segment(t_a: float, t_b: float):
        """Ventana [t_a, t_b) pedida al framework. Devuelve (ts, x, error)."""
        try:
            recs = eeg.get_data(start=t_a, end=t_b)
        except AssertionError as e:
            return None, None, str(e) or "AssertionError"
        recs = [r for r in recs if r.timestamp < t_b]
        if not recs:
            return None, None, "empty"
        ts = np.fromiter((r.timestamp for r in recs), dtype=float, count=len(recs))
        x = np.asarray([r.data for r in recs], dtype=np.float32).reshape(len(recs), n_ch)
        return ts, x, None

    def _decode(onset: float, ts, x):
        if ts is None or len(ts) < int(0.5 * sfreq):
            return None
        xf = filter_data(x.T.astype(np.float64) * 1e-6, sfreq, data.FMIN, data.FMAX,
                         method="iir", verbose=False)
        m = (ts >= onset + data.TMIN) & (ts < onset + data.TMAX)
        if m.sum() < int(0.5 * sfreq):
            return None
        proba = dec.predict_proba(xf[:, m][None])[0]
        pred = int(dec.model.classes_[int(np.argmax(proba))])
        return pred, float(proba.max()), int(m.sum())

    while True:
        now = local_clock()
        try:
            mk.get_latest_data()
            for r in new_markers:
                code = int(np.asarray(r.data).reshape(-1)[0])
                if code == 100:
                    state["t0"] = float(r.timestamp)
                elif code == 200:
                    state["ended_at"] = float(r.timestamp)
                elif code in data.EVENT_ID.values():
                    pending.append((float(r.timestamp), code))
            new_markers.clear()
            eeg.get_latest_data()
        except Exception:
            exc_total += 1
            if state["t0"] is not None:
                tel.on_exception(now, state["t0"])
            traceback.print_exc()
            time.sleep(0.05)
        t0, ended_at = state["t0"], state["ended_at"]

        still = []
        for onset, code in pending:
            t_end = onset + data.TMAX + PAD_S
            have = tel.last_ts is not None and tel.last_ts >= t_end
            timed_out = now > t_end + TRIAL_TIMEOUT_S
            if not (have or timed_out):
                still.append((onset, code)); continue
            if code not in DECODED:
                continue
            rts, rx, err = _segment(onset + data.TMIN - PAD_S, t_end)
            if err is not None and err != "empty":
                n_assert += 1
                exc_total += 1
                if t0 is not None:
                    tel.on_exception(now, t0)
            res = _decode(onset, rts, rx)
            t_dec = local_clock()
            k = len(trials)
            segments[f"t{k}_ts"] = (rts - onset).astype(np.float32) if rts is not None else np.empty(0, np.float32)
            segments[f"t{k}_x"] = rx if rx is not None else np.empty((0, n_ch), np.float32)
            if os.environ.get("BCIBENCH_DEBUG"):
                print(f"DBG trial onset_rel={onset - (t0 or 0):.3f} have={have} timed_out={timed_out} "
                      f"err={err} n={0 if rts is None else len(rts)} res={res}", flush=True)
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

        if ended_at is not None and not pending and now > ended_at + PAD_S + 1.0:
            break
        if t0 is not None and now > t0 + a.duration + TRIAL_TIMEOUT_S + PAD_S + 2.0 and not pending:
            break
        if now - state["last_data_wall"] > IDLE_ABORT_S and (t0 is None or now > t0 + 5):
            break
        if t0 is None and now - started_wall > RESOLVE_TIMEOUT:
            break
        time.sleep(POLL_S)

    n_buffered = len(eeg.buffer.get()) if eeg.buffer is not None else 0
    try:
        manager.stop_acquisition()
    except Exception:
        traceback.print_exc()

    t0, ended_at = state["t0"], state["ended_at"]
    with open(os.path.join(a.outdir, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["onset", "label", "pred", "proba", "n_samples", "valid", "decision_delay"])
        w.writeheader(); w.writerows(trials)
    if t0 is not None and ended_at is not None:
        dur_eff = ended_at - t0
    elif t0 is not None and tel.last_ts is not None:
        dur_eff = tel.last_ts - t0
    else:
        dur_eff = a.duration
    tel.dump(os.path.join(a.outdir, "telemetry.csv"), t0 or started_wall, dur_eff)
    meta = np.array([[t["onset"], t["label"], t["valid"]] for t in trials], dtype=np.float64).reshape(-1, 3)
    np.savez_compressed(os.path.join(a.outdir, "segments.npz"), meta=meta, sfreq=np.float64(sfreq),
                        tmin=np.float64(data.TMIN), tmax=np.float64(data.TMAX), pad=np.float64(PAD_S), **segments)
    valid = [t for t in trials if t["valid"]]
    y = np.array([t["label"] for t in valid]); p = np.array([t["pred"] for t in valid])
    bacc = None
    if len(valid):
        from sklearn.metrics import balanced_accuracy_score
        bacc = float(balanced_accuracy_score(y, p))
    summary = dict(exec_id=a.exec_id, t0=t0, duration_eff=dur_eff, n_trials=len(trials), n_valid=len(valid),
                   balanced_accuracy=bacc, exceptions=exc_total, n_samples_total=n_buffered,
                   consumer="bcipy", bcipy_version=version("bcipy"), pylsl_version=version("pylsl"),
                   max_buffer_len_s=MAX_BUFFER_S, max_buffer_samples=int(MAX_BUFFER_S * sfreq),
                   inlet_max_buflen_s=_bcipy_lsl_client.MAX_PAUSE_SECONDS,
                   get_data_assertions=n_assert)
    with open(os.path.join(a.outdir, "consumer.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
