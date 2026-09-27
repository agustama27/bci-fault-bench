"""Trazas de enlace real para el modo `trace` del inyector.

Una traza describe el estado de un enlace a lo largo del tiempo, en intervalos
consecutivos. Es el formato común al que se convierten mediciones publicadas de
distinta naturaleza (huecos en las marcas de tiempo de una grabación Bluetooth,
registros de ping sobre Wi-Fi congestionado, etc.); ver scripts/trace_tools.py.

Formato de archivo (CSV, UTF-8, `traces/<nombre>.csv`):

    # source: <de dónde sale la traza (URL, DOI, archivo)>
    # license: <licencia de la fuente>
    # resolution_s: <duración típica del intervalo>
    # notes: <cómo se convirtió; supuestos>
    t_s,up,delay_ms,loss[,jitter_ms]

    t_s        inicio del intervalo, en segundos desde el inicio de la traza (creciente)
    up         1 = enlace operativo; 0 = corte (el transporte se cae: se destruye el outlet)
    delay_ms   retraso adicional de entrega de los bloques emitidos en el intervalo (>= 0)
    loss       fracción de muestras omitidas en el intervalo (0..1)
    jitter_ms  (opcional) desviación estándar del retraso dentro del intervalo (>= 0)

El último intervalo dura `resolution_s` (o la mediana de los pasos si no se declara).

Anclaje a la escala del ensayo: la traza se reproduce 1:1 en el tiempo (un
segundo de traza es un segundo de corrida). No se estira ni se comprime: hacerlo
cambiaría la física del enlace. Si la traza es más corta que la corrida se
repite en forma cíclica. El punto de entrada (`offset`) se elige al azar a partir
de la semilla, salvo que se fije explícitamente (`<nombre>@<offset_s>`), para
que las corridas de un mismo sujeto no vean siempre el mismo tramo.

La severidad del modo `trace` es un factor de escala sobre `delay_ms` y `loss`
(1 = tal como se midió). Los cortes no se escalan.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRACE_DIR = os.environ.get("BCIBENCH_TRACES", os.path.join(ROOT, "traces"))
NAME_RE = re.compile(r"^(?P<name>[A-Za-z0-9_\-]+)(@(?P<offset>[0-9.]+))?$")


def parse_mode(mode: str) -> tuple[str, float | None]:
    """'bt-muse' -> ('bt-muse', None); 'bt-muse@120' -> ('bt-muse', 120.0)."""
    m = NAME_RE.match(mode or "")
    if not m:
        raise ValueError(f"nombre de traza inválido: {mode!r} (letras, dígitos, '-', '_' y opcional @offset)")
    off = m.group("offset")
    return m.group("name"), (float(off) if off is not None else None)


def trace_path(name: str) -> str:
    return name if os.path.isfile(name) else os.path.join(TRACE_DIR, f"{name}.csv")


@dataclass
class LinkTrace:
    t: np.ndarray            # inicio de cada intervalo (s)
    up: np.ndarray           # bool
    delay_s: np.ndarray      # s
    loss: np.ndarray         # 0..1
    jitter_s: np.ndarray     # s
    duration: float          # duración total de la traza (s)
    meta: dict = field(default_factory=dict)
    name: str = ""

    # ---------------------------------------------------------------- carga
    @classmethod
    def load(cls, name_or_path: str) -> "LinkTrace":
        path = trace_path(name_or_path)
        meta, rows = {}, []
        with open(path, encoding="utf-8") as f:
            header = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    k, _, v = line[1:].partition(":")
                    meta[k.strip()] = v.strip()
                    continue
                if header is None:
                    header = [h.strip() for h in line.split(",")]
                    continue
                rows.append([float(x) if x.strip() != "" else 0.0 for x in line.split(",")])
        if header is None or not rows:
            raise ValueError(f"traza vacía: {path}")
        col = {h: i for i, h in enumerate(header)}
        for req in ("t_s", "up", "delay_ms", "loss"):
            if req not in col:
                raise ValueError(f"la traza {path} no tiene la columna {req}")
        a = np.asarray(rows, dtype=float)
        t = a[:, col["t_s"]]
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"t_s debe ser estrictamente creciente en {path}")
        step = float(meta.get("resolution_s", np.median(np.diff(t)) if len(t) > 1 else 1.0))
        jit = a[:, col["jitter_ms"]] / 1e3 if "jitter_ms" in col else np.zeros(len(t))
        return cls(t=t - t[0], up=a[:, col["up"]] > 0.5,
                   delay_s=np.maximum(a[:, col["delay_ms"]], 0) / 1e3,
                   loss=np.clip(a[:, col["loss"]], 0, 1), jitter_s=np.maximum(jit, 0),
                   duration=float(t[-1] - t[0] + step), meta=meta,
                   name=os.path.splitext(os.path.basename(path))[0])

    # ---------------------------------------------------------------- consulta
    def index_at(self, t_rel: float, offset: float) -> int:
        tt = (t_rel + offset) % self.duration
        return int(np.searchsorted(self.t, tt, side="right") - 1)

    def state_at(self, t_rel: float, offset: float) -> tuple[bool, float, float, float]:
        """(up, delay_s, loss, jitter_s) del intervalo que cubre t_rel."""
        i = self.index_at(t_rel, offset)
        return bool(self.up[i]), float(self.delay_s[i]), float(self.loss[i]), float(self.jitter_s[i])

    def outages_within(self, duration_s: float, offset: float) -> list[tuple[float, float]]:
        """Cortes (inicio, fin) en tiempo de corrida, desenrollando la traza cíclica."""
        edges = np.append(self.t, self.duration)
        down = [(float(edges[i]), float(edges[i + 1])) for i in range(len(self.t)) if not self.up[i]]
        merged: list[list[float]] = []
        for s, e in down:
            if merged and abs(merged[-1][1] - s) < 1e-9:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        out = []
        k0 = int(np.floor(offset / self.duration))
        k1 = int(np.ceil((offset + duration_s) / self.duration)) + 1
        for k in range(k0, k1):
            for s, e in merged:
                a, b = s + k * self.duration - offset, e + k * self.duration - offset
                if b <= 0 or a >= duration_s:
                    continue
                out.append((max(a, 0.0), min(b, duration_s)))
        return sorted(out)

    # ---------------------------------------------------------------- resumen
    def summary(self) -> dict:
        edges = np.append(self.t, self.duration)
        w = np.diff(edges)
        upw = w[self.up]
        down = self.outages_within(self.duration, 0.0)
        dur = [b - a for a, b in down]
        d_up = self.delay_s[self.up] * 1e3
        return dict(
            name=self.name, duration_s=round(self.duration, 3), n_intervals=int(len(self.t)),
            resolution_s=round(float(np.median(w)), 4),
            up_fraction=round(float(upw.sum() / w.sum()), 4),
            n_outages=len(down),
            outage_s=dict(mean=round(float(np.mean(dur)), 3) if dur else 0.0,
                          max=round(float(max(dur)), 3) if dur else 0.0),
            loss_mean=round(float(np.average(self.loss[self.up], weights=upw)), 4) if upw.size else 0.0,
            delay_ms=dict(p50=round(float(np.percentile(d_up, 50)), 2) if upw.size else 0.0,
                          p99=round(float(np.percentile(d_up, 99)), 2) if upw.size else 0.0,
                          max=round(float(self.delay_s.max() * 1e3), 2)),
            jitter_ms_mean=round(float(np.mean(self.jitter_s[self.up] * 1e3)), 2) if upw.size else 0.0,
            meta=self.meta,
        )
