"""Telemetría del flujo por segundo nominal, compartida por los consumidores.

Vive aparte de consumer.py para que el consumidor sobre BciPy (otro entorno,
sin mne_lsl) la reutilice sin arrastrar dependencias.
"""
from __future__ import annotations

import csv

import numpy as np


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
