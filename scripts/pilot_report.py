"""Informe del piloto: insumos para fijar severidades, ventana W y regla del umbral.

Uso: python scripts/pilot_report.py --root results/piloto [--solo results/piloto-solo]

Imprime:
 1. Efecto de cada condición sobre infraestructura y desempeño (mediana entre sujetos).
 2. Para cada W y regla de umbral: fracción de ventanas de referencia marcadas como
    degradadas (falsas alarmas por construcción) y fracción marcada bajo cada tipo de fallo.
 3. Control de paralelismo: referencia con 6 ejecuciones simultáneas vs sola.
"""
import argparse
import warnings
warnings.filterwarnings("ignore")
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bcibench import metrics as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--solo", default=None)
a = ap.parse_args()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

execs, trials, tele = M.load_campaign(a.root)
print(f"\n{len(execs)} ejecuciones, sujetos {sorted(execs.subject.unique())}\n")

# 1. efecto por condición
cols = ["bacc", "valid_rate", "recv_ratio", "n_gaps", "lat_mean_ms", "ia_std_ms", "secs_no_data", "reconnections"]
med = execs.groupby(["kind", "severity", "label"])[cols].median().reset_index().sort_values(["kind", "severity"])
print("== 1. Efecto por condición (mediana entre sujetos)")
print(med.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n   bacc por sujeto y condición:")
print(execs.pivot_table(index="label", columns="subject", values="bacc").round(3).to_string())

# 2. ventana y umbral
print("\n== 2. Ventana móvil W y regla de umbral: fracción de ventanas (por segundo) marcadas como degradadas")
rows = []
for W in (6, 8, 12):
    rolls = {e: M.rolling_bacc(trials[e], W) for e in execs.exec_id}
    for rule, q in (("percentile", 5), ("percentile", 1), ("min", 0), ("mad", 2), ("mad", 3)):
        thr = {}
        for s in execs.subject.unique():
            refs = execs[(execs.subject == s) & (execs.kind == "none")].exec_id
            thr[s] = M.reference_threshold([rolls[e] for e in refs], q, rule)
        frac = {}
        for kind in ["none"] + M.KINDS:
            sub = execs[execs.kind == kind]
            vals = []
            for _, e in sub.iterrows():
                w = M.window_table(tele[e.exec_id], rolls[e.exec_id], thr[e.subject]).dropna(subset=["degraded"])
                vals.append(w["degraded"].mean() if len(w) else np.nan)
            frac[M.KIND_ES[kind]] = float(np.nanmean(vals)) if vals else np.nan
        rows.append(dict(W=W, regla=f"{rule}({q})", umbral_medio=float(np.nanmean(list(thr.values()))), **frac))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# 3. control de paralelismo
if a.solo and os.path.isdir(a.solo):
    solo, _, _ = M.load_campaign(a.solo)
    if len(solo):
        c = ["timing_p99_ms", "lat_mean_ms", "lat_max_ms", "ia_std_ms", "ia_max_ms", "recv_ratio", "bacc"]
        par = execs[execs.kind == "none"].set_index("subject")[c]
        so = solo.set_index("subject")[c]
        print("\n== 3. Control de paralelismo: referencia con 6 en paralelo (par) vs sola (solo)")
        print(pd.concat({"par": par, "solo": so}, axis=1).round(4).to_string())
print()
