"""Herramientas para trazas de enlace (modo `trace` del inyector; ver src/bcibench/trace.py).

Subcomandos:
  summary <traza>                      estadísticas de una traza (duración, cortes, pérdida, retraso)
  from-timestamps <csv> --sfreq 250    traza a partir de las marcas de tiempo de una grabación real
                                       (huecos entre muestras → pérdida; huecos largos → corte)
  from-arrivals <csv>                  traza de retraso/jitter a partir del instante de LLEGADA de cada
                                       muestra (p. ej. exportación de Mind Monitor de un Muse por BLE)
  from-ping <log>                      traza a partir de un registro de ping/RTT (una línea por paquete)
  synth-example                        traza sintética de EJEMPLO para pruebas de humo (no es una medición)

Uso:
  python scripts/trace_tools.py summary traces/ejemplo-sintetico.csv
  python scripts/trace_tools.py from-timestamps grabacion_ts.csv --column timestamp --sfreq 256 \
         --out traces/bt-casco-x.csv --source "URL/DOI" --license "CC-BY-4.0"
  python scripts/trace_tools.py from-ping ping.log --out traces/wifi-congestionado.csv --source ... --license ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bcibench.trace import LinkTrace  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def write_trace(path: str, t: np.ndarray, up: np.ndarray, delay_ms: np.ndarray, loss: np.ndarray,
                meta: dict, jitter_ms: np.ndarray | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        cols = "t_s,up,delay_ms,loss" + (",jitter_ms" if jitter_ms is not None else "")
        f.write(cols + "\n")
        for i in range(len(t)):
            row = f"{t[i]:.4f},{int(up[i])},{delay_ms[i]:.3f},{loss[i]:.4f}"
            if jitter_ms is not None:
                row += f",{jitter_ms[i]:.3f}"
            f.write(row + "\n")
    print(f"escrita {path}: {len(t)} intervalos, {t[-1] + float(meta.get('resolution_s', 0)):.1f} s")


# ---------------------------------------------------------------- from-timestamps
def from_timestamps(a):
    """Marcas de tiempo (s) de una grabación real → traza.

    Supuestos (declarados en # notes): la tasa nominal es --sfreq; un hueco entre muestras
    consecutivas mayor que 1,5 períodos son muestras perdidas; un hueco mayor que
    --outage-s es un corte del transporte. El retraso no es observable en las marcas de
    tiempo de la grabación (queda en 0) salvo que exista una columna de llegada (--arrival).
    """
    import pandas as pd
    df = pd.read_csv(a.path)
    ts = df[a.column].to_numpy(dtype=float)
    ts = ts - ts[0]
    period = 1.0 / a.sfreq
    step = a.resolution
    n_int = int(np.ceil(ts[-1] / step)) + 1
    t = np.arange(n_int) * step
    expected = np.full(n_int, step * a.sfreq)
    received = np.zeros(n_int)
    np.add.at(received, np.minimum((ts / step).astype(int), n_int - 1), 1)
    up = np.ones(n_int, dtype=bool)
    d = np.diff(ts)
    for i in np.flatnonzero(d > a.outage_s):
        s, e = ts[i] + period, ts[i + 1]
        up[int(s / step): int(np.ceil(e / step))] = False
    loss = np.clip(1.0 - received / expected, 0, 1)
    loss[~up] = 0.0
    delay_ms = np.zeros(n_int)
    jitter_ms = None
    if a.arrival:
        arr = df[a.arrival].to_numpy(dtype=float)
        lat = (arr - arr[0]) - ts                        # llegada − nominal, por muestra
        lat = lat - np.percentile(lat, 1)                # el retraso base no es observable: se toma el mínimo
        idx = np.minimum((ts / step).astype(int), n_int - 1)
        sums = np.zeros(n_int); sq = np.zeros(n_int)
        np.add.at(sums, idx, lat); np.add.at(sq, idx, lat ** 2)
        cnt = np.maximum(received, 1)
        mean = sums / cnt
        delay_ms = np.maximum(mean, 0) * 1e3
        jitter_ms = np.sqrt(np.maximum(sq / cnt - mean ** 2, 0)) * 1e3
    meta = dict(source=a.source, license=a.license, resolution_s=step,
                notes=(f"from-timestamps: sfreq nominal {a.sfreq} Hz; muestra perdida = hueco > 1,5 períodos; "
                       f"corte = hueco > {a.outage_s} s; retraso "
                       + ("estimado de la columna de llegada (base = percentil 1)" if a.arrival else "no observable (0)")))
    write_trace(a.out, t, up, delay_ms, loss, meta, jitter_ms)


# ---------------------------------------------------------------- from-arrivals
def from_arrivals(a):
    """Instantes de LLEGADA por muestra de una grabación real (p. ej. Mind Monitor: hora del
    teléfono en que llegó cada muestra del Muse por BLE) → traza de retraso y jitter.

    La grabación no trae el instante nominal de cada muestra: se estima un reloj lineal por
    mínimos cuadrados sobre (índice, llegada) (tasa efectiva del dispositivo) y el residuo
    llegada − nominal es el retraso de entrega, referido a su percentil 1 (el retraso base no
    es observable). Por intervalo: delay_ms = media del residuo, jitter_ms = desvío. La
    pérdida NO es observable en esta fuente (loss = 0) y no hay cortes (up = 1), salvo que
    un hueco entre llegadas supere --outage-s. Supuestos declarados en # notes.
    """
    import pandas as pd
    df = pd.read_csv(a.path, usecols=[a.column] + ([a.presence] if a.presence else []))
    if a.presence:
        df = df[df[a.presence].notna()]
    col = df[a.column]
    if a.datetime:
        ts = pd.to_datetime(col)
        arr = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    else:
        arr = col.to_numpy(dtype=float); arr = arr - arr[0]
    i = np.arange(len(arr))
    slope, intercept = np.polyfit(i, arr, 1)
    res = arr - (intercept + slope * i)
    base = np.percentile(res, 1)
    lat = np.maximum(res - base, 0.0)
    step = a.resolution
    n_int = int(np.ceil(arr[-1] / step)) + 1
    t = np.arange(n_int) * step
    idx = np.minimum((arr / step).astype(int), n_int - 1)
    cnt = np.bincount(idx, minlength=n_int).astype(float)
    sums = np.bincount(idx, weights=lat, minlength=n_int)
    sq = np.bincount(idx, weights=lat ** 2, minlength=n_int)
    mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    std = np.sqrt(np.maximum(np.where(cnt > 0, sq / np.maximum(cnt, 1), 0) - np.nan_to_num(mean) ** 2, 0))
    # intervalos sin llegadas: heredan el último valor conocido (el enlace sigue entregando tarde)
    for k in range(1, n_int):
        if np.isnan(mean[k]):
            mean[k], std[k] = mean[k - 1], std[k - 1]
    mean = np.nan_to_num(mean)
    up = np.ones(n_int, dtype=bool)
    d = np.diff(arr)
    for j in np.flatnonzero(d > a.outage_s):
        up[int(arr[j] / step) + 1: int(np.ceil(arr[j + 1] / step))] = False
    meta = dict(source=a.source, license=a.license, resolution_s=step,
                notes=(f"from-arrivals: {len(arr)} llegadas en {arr[-1]:.1f} s; reloj nominal ajustado por mínimos "
                       f"cuadrados ({1 / slope:.2f} Hz); retraso = residuo llegada - nominal referido a su percentil 1 "
                       f"({base * 1e3:.1f} ms); jitter = desvío del residuo por intervalo; la pérdida no es observable "
                       f"en esta fuente (loss = 0); corte = hueco entre llegadas > {a.outage_s} s"))
    write_trace(a.out, t, up, mean * 1e3, np.zeros(n_int), meta, std * 1e3)


# ---------------------------------------------------------------- from-ping
PING_RE = re.compile(r"icmp_seq=(\d+).*?time=([\d.]+)\s*ms", re.I)


def from_ping(a):
    """Registro de `ping` (Linux/Windows) o CSV seq,rtt_ms (vacío = perdido) → traza.

    Supuestos: un paquete por --interval s; retraso de un sentido = RTT/2 (declarado);
    paquete perdido → intervalo con loss = 1; --outage-n paquetes perdidos seguidos → corte.
    """
    seq, rtt = [], []
    if a.path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(a.path)
        seq = df.iloc[:, 0].to_numpy(dtype=int)
        rtt = df.iloc[:, 1].to_numpy(dtype=float)
    else:
        for line in open(a.path, encoding="utf-8", errors="ignore"):
            m = PING_RE.search(line)
            if m:
                seq.append(int(m.group(1))); rtt.append(float(m.group(2)))
        seq, rtt = np.asarray(seq), np.asarray(rtt)
        # secuencias faltantes = paquetes perdidos
        full = np.arange(seq.min(), seq.max() + 1)
        r = np.full(len(full), np.nan)
        r[seq - seq.min()] = rtt
        seq, rtt = full, r
    n = len(seq)
    t = np.arange(n) * a.interval
    lost = np.isnan(rtt)
    up = np.ones(n, dtype=bool)
    # corridas de pérdidas consecutivas >= outage-n → corte
    i = 0
    while i < n:
        if lost[i]:
            j = i
            while j < n and lost[j]:
                j += 1
            if j - i >= a.outage_n:
                up[i:j] = False
            i = j
        else:
            i += 1
    base = np.nanpercentile(rtt, 1) if a.subtract_base else 0.0
    one_way = np.where(lost, 0.0, np.maximum(rtt - base, 0) / 2.0)
    loss = np.where(lost & up, 1.0, 0.0)
    meta = dict(source=a.source, license=a.license, resolution_s=a.interval,
                notes=(f"from-ping: un paquete cada {a.interval} s; retraso de un sentido = (RTT − base)/2 con "
                       f"base = percentil 1 ({base:.2f} ms)" if a.subtract_base else "RTT/2") +
                      f"; paquete perdido = loss 1; >= {a.outage_n} pérdidas seguidas = corte")
    write_trace(a.out, t, up, one_way, loss, meta)


# ---------------------------------------------------------------- synth-example
def synth_example(a):
    """Traza sintética de ejemplo (60 s, paso 0,1 s): NO es una medición.

    Sirve para probar el modo trace de punta a punta: 10 s limpios, 10 s con retraso
    creciente 0→60 ms y jitter 5 ms, un corte de 0,8 s, 10 s con pérdida 3 %, resto limpio.
    """
    step = 0.1
    t = np.arange(0, 60, step)
    n = len(t)
    up = np.ones(n, dtype=bool)
    delay = np.zeros(n); loss = np.zeros(n); jit = np.zeros(n)
    m = (t >= 10) & (t < 20); delay[m] = np.linspace(0, 60, m.sum()); jit[m] = 5
    up[(t >= 25) & (t < 25.8)] = False
    loss[(t >= 30) & (t < 40)] = 0.03
    meta = dict(source="sintética (scripts/trace_tools.py synth-example)", license="MIT (este repositorio)",
                resolution_s=step,
                notes="EJEMPLO para pruebas de humo; no representa ningún enlace medido")
    write_trace(a.out, t, up, delay, loss, meta, jit)


def summary(a):
    print(json.dumps(LinkTrace.load(a.path).summary(), indent=2, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary"); s.add_argument("path"); s.set_defaults(fn=summary)
    s = sub.add_parser("from-timestamps"); s.add_argument("path")
    s.add_argument("--column", default="timestamp"); s.add_argument("--arrival", default=None)
    s.add_argument("--sfreq", type=float, required=True); s.add_argument("--resolution", type=float, default=0.1)
    s.add_argument("--outage-s", type=float, default=0.5)
    s.add_argument("--out", required=True); s.add_argument("--source", required=True); s.add_argument("--license", required=True)
    s.set_defaults(fn=from_timestamps)
    s = sub.add_parser("from-arrivals"); s.add_argument("path")
    s.add_argument("--column", default="TimeStamp", help="columna con el instante de llegada")
    s.add_argument("--datetime", action="store_true", help="la columna es fecha-hora (no segundos)")
    s.add_argument("--presence", default=None, help="solo filas con esta columna no vacía (p. ej. RAW_TP9)")
    s.add_argument("--resolution", type=float, default=0.1); s.add_argument("--outage-s", type=float, default=0.5)
    s.add_argument("--out", required=True); s.add_argument("--source", required=True); s.add_argument("--license", required=True)
    s.set_defaults(fn=from_arrivals)
    s = sub.add_parser("from-ping"); s.add_argument("path")
    s.add_argument("--interval", type=float, default=0.2); s.add_argument("--outage-n", type=int, default=3)
    s.add_argument("--subtract-base", action="store_true", default=True)
    s.add_argument("--out", required=True); s.add_argument("--source", required=True); s.add_argument("--license", required=True)
    s.set_defaults(fn=from_ping)
    s = sub.add_parser("synth-example"); s.add_argument("--out", default=os.path.join(ROOT, "traces", "ejemplo-sintetico.csv"))
    s.set_defaults(fn=synth_example)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
