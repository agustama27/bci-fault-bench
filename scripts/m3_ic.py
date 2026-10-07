"""M3 · Tarea 3: IC 95 % de la diferencia apareada (condición - referencia) en balanced accuracy.

Unidad = sujeto (n = 9): mediana de las 5 corridas, tal cual analysis/medianas_por_sujeto.csv
(CSP+LDA en línea) y analysis-eegnet/medianas_por_sujeto.csv (EEGNet).

Métodos
  Estimador   : Hodges-Lehmann (mediana de los 45 promedios de Walsh de las 9 diferencias).
  IC principal: intervalo exacto basado en el estadístico de rangos con signo de Wilcoxon, a partir de los
                promedios de Walsh ordenados: [W_(k+1), W_(45-k)], k = mayor entero con P(T+ <= k) <= 0,025
                bajo H0 (distribución exacta por enumeración de los 2^9 signos). Con n = 9, k = 5 y el nivel
                de confianza real es 1 - 2 P(T+ <= 5) = 96,1 %.
  Contraste   : bootstrap por sujetos (10 000 réplicas, semilla 20261007), percentil 2,5 - 97,5 de HL.
  Equivalencia: se marca "dentro" cuando el IC exacto queda contenido en [-0,03; +0,03].
Salida: campaign-2026-09-27/analysis-m3/m3_ic.csv
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

ROOT = r"C:\Users\agustin.tamagusuku\Desktop\bci-fault-bench\campaign-2026-09-27"
OUT = os.path.join(ROOT, "analysis-m3")
SEED = 20261007
B = 10_000
MARGIN = 0.03
LABELS = ["loss-random-0.01", "loss-random-0.05", "loss-random-0.1", "jitter-0.01", "jitter-0.05", "jitter-0.1",
          "delay-0.05", "delay-0.1", "delay-0.25", "disconnect-0.5", "disconnect-1", "disconnect-3",
          "burst_trial-0.1", "burst_trial-0.25", "burst_trial-0.4",
          "disconnect_trial-0.5", "disconnect_trial-1", "disconnect_trial-2"]


def walsh(d: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(len(d))
    return (d[i] + d[j]) / 2.0


def signed_rank_cdf(n: int):
    """P(T+ <= k) bajo H0, k = 0..n(n+1)/2, por programación dinámica."""
    maxs = n * (n + 1) // 2
    cnt = np.zeros(maxs + 1)
    cnt[0] = 1
    for r in range(1, n + 1):
        new = cnt.copy()
        new[r:] += cnt[:maxs + 1 - r]
        cnt = new
    return np.cumsum(cnt) / 2.0 ** n


def exact_ci(d: np.ndarray, alpha: float = 0.05):
    n = len(d)
    cdf = signed_rank_cdf(n)
    k = int(np.max(np.where(cdf <= alpha / 2)[0]))
    w = np.sort(walsh(d))
    m = len(w)
    lo, hi = w[k], w[m - k - 1]          # W_(k+1), W_(m-k)  (índices base 0)
    conf = 1 - 2 * cdf[k]
    return float(lo), float(hi), float(conf), k


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for dec, folder in (("CSP+LDA", "analysis"), ("EEGNet", "analysis-eegnet")):
        med = pd.read_csv(os.path.join(ROOT, folder, "medianas_por_sujeto.csv"))
        ref = med[med.label == "ref"].set_index("subject").bacc.sort_index()
        for lab in LABELS:
            s = med[med.label == lab].set_index("subject").bacc.sort_index()
            d = (s - ref).dropna().to_numpy(float)
            n = len(d)
            hl = float(np.median(walsh(d)))
            lo, hi, conf, k = exact_ci(d)
            idx = rng.integers(0, n, size=(B, n))
            samp = d[idx]
            i, j = np.triu_indices(n)
            wal = (samp[:, i] + samp[:, j]) / 2.0
            hl_b = np.median(wal, axis=1)
            blo, bhi = np.percentile(hl_b, [2.5, 97.5])
            rows.append(dict(decoder=dec, label=lab, n=n, mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                             n_neg=int((d < 0).sum()), n_zero=int((d == 0).sum()), n_pos=int((d > 0).sum()),
                             HL=hl, ci_lo=lo, ci_hi=hi, conf_real=conf,
                             boot_lo=float(blo), boot_hi=float(bhi),
                             dentro_exacto=bool(lo >= -MARGIN - 1e-12 and hi <= MARGIN + 1e-12),
                             dentro_boot=bool(blo >= -MARGIN - 1e-12 and bhi <= MARGIN + 1e-12),
                             ancho_exacto=hi - lo))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "m3_ic.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("nivel de confianza real del IC exacto (n=9):", round(R.conf_real.iloc[0], 4), "k =", exact_ci(np.arange(9.0))[3])
    print(R.round(4).to_string())
    print("\ndentro de +-0,03 (IC exacto):")
    for dec in ("CSP+LDA", "EEGNet"):
        g = R[R.decoder == dec]
        print(dec, "dentro:", list(g[g.dentro_exacto].label), "\n   fuera:", list(g[~g.dentro_exacto].label))


if __name__ == "__main__":
    main()
