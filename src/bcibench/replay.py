"""Servicio de reproducción: emite una corrida como flujo LSL a su tasa original,
con el inyector de fallos aplicado en la interfaz de transporte.

Dos flujos LSL por ejecución:
  <exec>-eeg      22 canales float32 (µV) a 250 Hz   ← aquí actúan los fallos
  <exec>-markers  int32 irregular: 100 = inicio (t0); 1..4 = inicio de ensayo

Los marcadores NO se perturban: representan al software de estímulo, no al
transporte de señal. Las marcas de tiempo LSL de cada muestra son las
nominales (t0 + i/sfreq), como haría un amplificador real; un retraso se ve en
el consumidor como diferencia entre llegada y marca de tiempo.

Uso: python -m bcibench.replay --subject 1 --session 1test --run 0 \
        --exec-id X --fault loss --severity 0.05 --seed 7 --outdir results/raw/X
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock

from . import data
from .injector import FaultSpec, Injector

CHUNK = 10                 # muestras por bloque (40 ms a 250 Hz)
START_MARKER = 100
END_MARKER = 200
CONSUMER_WAIT_S = 60.0


def _sleep_until(t: float):
    """Espera híbrida: sleep grueso + espera activa los últimos 2 ms."""
    while True:
        rem = t - local_clock()
        if rem <= 0:
            return
        if rem > 0.002:
            time.sleep(rem - 0.002)
        else:
            pass  # espera activa


def _eeg_info(exec_id: str, n_ch: int, sfreq: float, ch_names: list[str]) -> StreamInfo:
    info = StreamInfo(f"{exec_id}-eeg", "EEG", n_ch, sfreq, "float32", f"{exec_id}-eeg")
    info.set_channel_names(ch_names)
    info.set_channel_types("eeg")
    info.set_channel_units("microvolts")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--session", default=data.SESSION_TEST)
    ap.add_argument("--run", default="0")
    ap.add_argument("--exec-id", required=True)
    ap.add_argument("--fault", default="none")
    ap.add_argument("--severity", type=float, default=0.0)
    ap.add_argument("--mode", default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-seconds", type=float, default=None, help="solo desarrollo: trunca la corrida")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    raw = data.load_subject(a.subject)[a.session][a.run]
    picks = data.eeg_picks(raw)
    X = (raw.get_data(picks=picks) * 1e6).astype(np.float32).T     # (n_samples, n_ch)
    ch_names = [raw.ch_names[i] for i in picks]
    sfreq = float(raw.info["sfreq"])
    events = data.events_of(raw)                                    # (n, 3) muestra absoluta
    ev_samples = events[:, 0] - raw.first_samp
    ev_codes = events[:, 2]
    n_total = len(X)
    if a.max_seconds:
        n_total = min(n_total, int(a.max_seconds * sfreq))
    duration = n_total / sfreq

    spec = FaultSpec(a.fault, a.severity, a.mode, a.seed)
    inj = Injector(spec, sfreq, duration, trial_onsets_samples=ev_samples[ev_samples < n_total])

    einfo = _eeg_info(a.exec_id, len(ch_names), sfreq, ch_names)
    outlet = StreamOutlet(einfo, chunk_size=CHUNK)
    minfo = StreamInfo(f"{a.exec_id}-markers", "Markers", 1, 0.0, "int32", f"{a.exec_id}-markers")
    mout = StreamOutlet(minfo)
    if not outlet.wait_for_consumers(CONSUMER_WAIT_S) or not mout.wait_for_consumers(CONSUMER_WAIT_S):
        print("replay: sin consumidor, abortando", file=sys.stderr)
        sys.exit(2)

    t0 = local_clock() + 1.0
    mout.push_sample(np.array([START_MARKER], dtype=np.int32), timestamp=t0)

    log = open(os.path.join(a.outdir, "fault_log.jsonl"), "w")
    for s, e in inj.outages:
        log.write(json.dumps({"event": "outage_planned", "start": s, "end": e}) + "\n")
    for i_a, i_b in inj.drop_intervals:
        log.write(json.dumps({"event": "interval_planned", "start_sample": int(i_a), "end_sample": int(i_b)}) + "\n")

    n_pushed = n_dropped = 0
    timing_err = []
    ev_i = 0
    connected = True
    reconnections = 0
    n_chunks = int(np.ceil(n_total / CHUNK))
    for c in range(n_chunks):
        i0 = c * CHUNK
        i1 = min(i0 + CHUNK, n_total)
        t_nom = t0 + i0 / sfreq
        # marcadores de ensayo que caen en este bloque, a su hora nominal
        while ev_i < len(ev_samples) and ev_samples[ev_i] < i1:
            ts = t0 + ev_samples[ev_i] / sfreq
            _sleep_until(ts)
            mout.push_sample(np.array([int(ev_codes[ev_i])], dtype=np.int32), timestamp=ts)
            ev_i += 1
        # Un amplificador entrega el bloque cuando adquirió su última muestra:
        # el instante nominal de entrega es el de la última muestra del bloque.
        t_last = t0 + (i1 - 1) / sfreq
        target = t_last + inj.extra_delay()
        _sleep_until(target)
        timing_err.append(local_clock() - target)
        t_rel = i0 / sfreq
        if inj.in_outage(t_rel):
            if connected:
                del outlet
                gc.collect()
                connected = False
                log.write(json.dumps({"event": "outage_start", "t_rel": t_rel}) + "\n")
            n_dropped += i1 - i0
            continue
        if not connected:
            outlet = StreamOutlet(einfo, chunk_size=CHUNK)
            connected = True
            reconnections += 1
            log.write(json.dumps({"event": "outage_end", "t_rel": t_rel}) + "\n")
        chunk = inj.transform(X[i0:i1], i0)
        keep = inj.keep_mask(len(chunk), i0)
        if keep.any():
            ts = t_nom + np.flatnonzero(keep) / sfreq
            outlet.push_chunk(chunk[keep], timestamp=ts)
        n_pushed += int(keep.sum())
        n_dropped += int((~keep).sum())

    # marcador de fin a su hora nominal; deja que el consumidor drene
    t_end = t0 + n_total / sfreq
    _sleep_until(t_end)
    mout.push_sample(np.array([END_MARKER], dtype=np.int32), timestamp=t_end)
    time.sleep(2.0)
    te = np.array(timing_err)
    summary = {
        "exec_id": a.exec_id, "subject": a.subject, "session": a.session, "run": a.run,
        "fault": spec.to_dict(), "label": spec.label(), "t0": t0, "sfreq": sfreq,
        "n_channels": len(ch_names), "n_total": n_total, "duration_s": duration,
        "n_pushed": n_pushed, "n_dropped": n_dropped, "outages": inj.outages,
        "reconnections": reconnections, "n_events": int(ev_i),
        "timing_err_ms": {"mean": float(te.mean() * 1e3), "p50": float(np.median(te) * 1e3),
                          "p99": float(np.percentile(te, 99) * 1e3), "max": float(te.max() * 1e3)},
    }
    log.close()
    with open(os.path.join(a.outdir, "producer.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("label", "n_pushed", "n_dropped", "reconnections", "timing_err_ms")}))


if __name__ == "__main__":
    main()
