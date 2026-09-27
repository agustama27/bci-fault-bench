"""Modelos de fallo sobre la interfaz de transporte (Tabla 1 de Métodos).

kind        dónde actúa                       severidad
----------  --------------------------------  ------------------------------
none        —                                 —
loss        contenido del flujo               fracción de muestras omitidas (0.01/0.05/0.10)
jitter      instante de entrega de cada bloque desviación estándar en s (0.010/0.050/0.100)
delay       instante de entrega de cada bloque desplazamiento constante en s (0.05/0.10/0.25)
disconnect  transporte real (outlet se cierra) duración de cada corte en s (0.5/1/3)

Toda aleatoriedad sale de una semilla fija: la realización del fallo es
repetible (Métodos: "una realización del fallo fijada por una semilla").
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

BURST_LEN = 25            # muestras por ráfaga (100 ms a 250 Hz)
N_DISCONNECTS = 5         # cortes por corrida, en instantes aleatorios
GUARD_S = 15.0            # sin cortes en los primeros/últimos segundos


@dataclass
class FaultSpec:
    kind: str = "none"          # none | loss | jitter | delay | disconnect
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
    def __init__(self, spec: FaultSpec, sfreq: float, duration_s: float):
        self.spec = spec
        self.sfreq = sfreq
        self.rng = np.random.default_rng(spec.seed)
        self.outages: list[tuple[float, float]] = []   # (inicio, fin) en s relativos a t0
        self._burst_left = 0                            # muestras de ráfaga que continúan en el próximo bloque
        if spec.kind == "disconnect":
            self.outages = self._plan_outages(duration_s, spec.severity)

    # --- planificación -------------------------------------------------
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

    # --- aplicación por bloque -----------------------------------------
    def keep_mask(self, n: int) -> np.ndarray:
        """Máscara de muestras que sobreviven en un bloque de n muestras."""
        if self.spec.kind != "loss":
            return np.ones(n, dtype=bool)
        p = self.spec.severity
        if self.spec.mode == "random":
            return self.rng.random(n) >= p
        # ráfagas: cada muestra inicia una ráfaga con prob p/BURST_LEN; una
        # ráfaga que excede el bloque continúa en el siguiente (fracción esperada = p)
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
