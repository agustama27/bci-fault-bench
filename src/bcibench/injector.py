"""Modelos de fallo sobre la interfaz de transporte.

Familia 1 (Tabla 1 del Entregable 1): distribución uniforme en el tiempo.
kind        dónde actúa                        severidad
----------  ---------------------------------  ------------------------------
none        —                                  —
loss        contenido del flujo                fracción de muestras omitidas (0.01/0.05/0.10)
jitter      instante de entrega de cada bloque desviación estándar en s (0.010/0.050/0.100)
delay       instante de entrega de cada bloque desplazamiento constante en s (0.05/0.10/0.25)
disconnect  transporte real (outlet se cierra) duración de cada corte en s (0.5/1/3), 5 cortes por corrida

Familia 2 (segunda campaña): fallos ESTRUCTURADOS, sincronizados con el ensayo.
Misma escala del ensayo (ventana de decodificación de 4 s) como criterio de severidad.
kind              dónde actúa                     severidad
----------------  ------------------------------  ---------------------------------------------
burst_trial       contenido del flujo             fracción de la ventana del ensayo borrada en un
                                                  solo bloque contiguo (0.10/0.25/0.40 = 0,4/1/1,6 s),
                                                  ubicado al azar dentro de la ventana; todos los ensayos
disconnect_trial  transporte real                 corte que comienza al inicio de la ventana del
                                                  ensayo y dura 0.5/1/2 s; todos los ensayos

Toda aleatoriedad sale de una semilla fija: la realización del fallo es repetible.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

BURST_LEN = 25            # muestras por ráfaga (100 ms a 250 Hz), modelo loss-burst
N_DISCONNECTS = 5         # cortes por corrida, familia 1
GUARD_S = 15.0            # sin cortes en los primeros/últimos segundos
TRIAL_TMIN, TRIAL_TMAX = 2.0, 6.0   # ventana de decodificación relativa al inicio del ensayo
STRUCTURED = ("burst_trial", "disconnect_trial")


@dataclass
class FaultSpec:
    kind: str = "none"          # none | loss | jitter | delay | disconnect | burst_trial | disconnect_trial
    severity: float = 0.0
    mode: str = "random"        # loss: random | burst
    seed: int = 0

    def label(self) -> str:
        if self.kind == "none":
            return "ref"
        m = f"-{self.mode}" if self.kind == "loss" else ""
        return f"{self.kind}{m}-{self.severity:g}"

    def to_dict(self) -> dict:
        return asdict(self)


class Injector:
    def __init__(self, spec: FaultSpec, sfreq: float, duration_s: float,
                 trial_onsets_samples=None):
        self.spec = spec
        self.sfreq = sfreq
        self.rng = np.random.default_rng(spec.seed)
        self.outages: list[tuple[float, float]] = []   # (inicio, fin) en s relativos a t0
        self.drop_intervals: list[tuple[int, int]] = []  # [i0, i1) en muestras, familia 2
        self._burst_left = 0
        onsets = np.asarray(trial_onsets_samples if trial_onsets_samples is not None else [], dtype=int)
        if spec.kind == "disconnect":
            self.outages = self._plan_outages(duration_s, spec.severity)
        elif spec.kind == "burst_trial":
            win = int(round((TRIAL_TMAX - TRIAL_TMIN) * sfreq))
            length = int(round(spec.severity * win))
            for on in onsets:
                w0 = on + int(round(TRIAL_TMIN * sfreq))
                start = w0 + int(self.rng.integers(0, max(win - length, 0) + 1))
                self.drop_intervals.append((start, start + length))
        elif spec.kind == "disconnect_trial":
            for on in onsets:
                t = on / sfreq + TRIAL_TMIN
                self.outages.append((float(t), float(t + spec.severity)))

    # --- planificación (familia 1) ----------------------------------------
    def _plan_outages(self, duration_s: float, dur: float) -> list[tuple[float, float]]:
        lo, hi = GUARD_S, max(GUARD_S + 1, duration_s - GUARD_S - dur)
        starts = np.sort(self.rng.uniform(lo, hi, N_DISCONNECTS))
        out, last_end = [], -1.0
        for s in starts:
            s = max(s, last_end + 5.0)             # al menos 5 s entre cortes
            if s + dur > duration_s - GUARD_S:
                break
            out.append((float(s), float(s + dur)))
            last_end = s + dur
        return out

    # --- aplicación por bloque --------------------------------------------
    def keep_mask(self, n: int, i0: int = 0) -> np.ndarray:
        """Máscara de muestras que sobreviven en el bloque [i0, i0+n)."""
        k = self.spec.kind
        if k == "loss":
            p = self.spec.severity
            if self.spec.mode == "random":
                return self.rng.random(n) >= p
            mask = np.ones(n, dtype=bool)
            draws = self.rng.random(n)
            for i in range(n):
                if self._burst_left > 0:
                    mask[i] = False
                    self._burst_left -= 1
                elif draws[i] < p / BURST_LEN:
                    mask[i] = False
                    self._burst_left = BURST_LEN - 1
            return mask
        if k == "burst_trial":
            mask = np.ones(n, dtype=bool)
            idx = np.arange(i0, i0 + n)
            for a, b in self.drop_intervals:
                if b <= i0 or a >= i0 + n:
                    continue
                mask[(idx >= a) & (idx < b)] = False
            return mask
        return np.ones(n, dtype=bool)

    def extra_delay(self) -> float:
        """Segundos que se suman al instante nominal de entrega del bloque."""
        k = self.spec.kind
        if k == "delay":
            return self.spec.severity
        if k == "jitter":
            return float(max(0.0, self.rng.normal(0.0, self.spec.severity)))
        return 0.0

    def in_outage(self, t_rel: float) -> bool:
        return any(a <= t_rel < b for a, b in self.outages)
