"""Convierte las salidas de analyze.py en tablas para contenido.js (formato APA,
decimales con coma) y copia las figuras a la carpeta del manuscrito.

Uso: python scripts/resultados_tablas.py --analysis results/analysis --manuscrito ../30-TFG/Entregas/Modulo-2
Salida: <manuscrito>/src/tablas.json  y  <manuscrito>/src/fig/*.png
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--analysis", required=True)
ap.add_argument("--manuscrito", required=True)
a = ap.parse_args()
A = a.analysis
out_src = os.path.join(a.manuscrito, "src")
fig_dir = os.path.join(out_src, "fig")
os.makedirs(fig_dir, exist_ok=True)

KIND_ES = {"none": "Referencia", "loss": "Pérdida de muestras", "jitter": "*Jitter*", "delay": "Retraso", "disconnect": "Desconexión"}
UNIT = {"loss": lambda s: f"{s*100:g} %".replace(".", ","), "jitter": lambda s: f"{s*1000:g} ms".replace(".", ","), "delay": lambda s: f"{s*1000:g} ms".replace(".", ","), "disconnect": lambda s: f"{s:g} s".replace(".", ",")}


def f(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def sev(kind, s):
    return UNIT[kind](s) if kind in UNIT else "—"


def star(p):
    return "*" if (p is not None and not np.isnan(p) and p < 0.05) else ""


tablas = {}

# --- Tabla desempeño (obj. 3): bacc por condición, diferencia, p Holm, r
desc = pd.read_csv(os.path.join(A, "descriptivos.csv"))
t_b = pd.read_csv(os.path.join(A, "t_desempeno.csv"))
rows = []
ref = desc[desc.kind == "none"].iloc[0]
rows.append(["Referencia", "—", f(ref.bacc, 3), f(ref.valid_rate, 3), "—", "—", "—"])
for kind in ["loss", "jitter", "delay", "disconnect"]:
    for _, r in t_b[t_b.kind == kind].sort_values("severity").iterrows():
        d = desc[(desc.kind == kind) & (desc.severity == r.severity)].iloc[0]
        rows.append([KIND_ES[kind], sev(kind, r.severity), f(d.bacc, 3), f(d.valid_rate, 3),
                     f(r["diff"], 3), f(r.p_holm, 3) + star(r.p_holm), f(r.r, 2)])
tablas["desempeno"] = dict(
    headers=["Tipo de fallo", "Severidad", "*Balanced accuracy*", "Decisiones válidas", "Diferencia vs. referencia", "*p* (Holm)", "*r*"],
    rows=rows,
    friedman={k: (None if np.isnan(g.friedman_chi2.iloc[0]) else float(g.friedman_chi2.iloc[0]), None if np.isnan(g.friedman_p.iloc[0]) else float(g.friedman_p.iloc[0]), int(g.n.iloc[0])) for k, g in t_b.groupby("kind")},
)

# --- Tabla infraestructura (obj. 2)
rows = []
rows.append(["Referencia", "—", f(ref.recv_ratio, 3), f(ref.n_gaps, 0), f(ref.lat_mean_ms, 1), f(ref.ia_std_ms, 1), f(ref.secs_no_data, 0)])
t_i = pd.read_csv(os.path.join(A, "t_infra.csv"))
for kind in ["loss", "jitter", "delay", "disconnect"]:
    for s in sorted(desc[desc.kind == kind].severity.unique()):
        d = desc[(desc.kind == kind) & (desc.severity == s)].iloc[0]
        def p_of(metric):
            q = t_i[(t_i.metric == metric) & (t_i.kind == kind) & (t_i.severity == s)]
            return star(float(q.p_holm.iloc[0])) if len(q) else ""
        rows.append([KIND_ES[kind], sev(kind, s), f(d.recv_ratio, 3) + p_of("recv_ratio"), f(d.n_gaps, 0) + p_of("n_gaps"),
                     f(d.lat_mean_ms, 1) + p_of("lat_mean_ms"), f(d.ia_std_ms, 1) + p_of("ia_std_ms"),
                     f(d.secs_no_data, 0) + p_of("secs_no_data")])
tablas["infra"] = dict(
    headers=["Tipo de fallo", "Severidad", "Muestras recibidas / esperadas", "Huecos", "Latencia media (ms)", "Desv. intervalo entre bloques (ms)", "Segundos sin datos"],
    rows=rows)

# --- Tabla divergencia (obj. 4)
t_d = pd.read_csv(os.path.join(A, "t_divergencia.csv"))
rows = []
for _, r in t_d.iterrows():
    rows.append([KIND_ES[r.kind], sev(r.kind, r.severity) if r.kind != "none" else "—", f(r.silent, 3), f(r.loud, 3), f(r.divergent, 3)])
tablas["divergencia"] = dict(headers=["Tipo de fallo", "Severidad", "Operativo y degradado", "No operativo y no degradado", "Total divergentes"], rows=rows)

# --- Tabla detectores (obj. 5)
t_det = pd.read_csv(os.path.join(A, "t_detectores.csv"))
NAMES = {"umbrales": "Umbrales", "logistic": "Regresión logística", "random_forest": "*Random forest*", "gradient_boosting": "*Gradient boosting*"}
rows = [[NAMES.get(r.detector, r.detector), f(r.precision, 3), f(r.recall, 3), f(r.f1, 3), f(r.fpr, 3), f(r.detection_delay_s, 1), f"{int(r.episodes_detected)}/{int(r.episodes)}"] for _, r in t_det.iterrows()]
tablas["detectores"] = dict(headers=["Detector", "Precisión", "Sensibilidad", "F1", "Tasa de falsos positivos", "Retardo de detección (s)", "Episodios detectados"], rows=rows)

# --- umbrales y ventana
thr = json.load(open(os.path.join(A, "thresholds.json")))
tablas["umbrales"] = thr

with open(os.path.join(out_src, "tablas.json"), "w", encoding="utf-8") as fh:
    json.dump(tablas, fh, ensure_ascii=False, indent=1)
for fn in ("fig_degradacion.png", "fig_latencia.png", "fig_integridad.png"):
    if os.path.exists(os.path.join(A, fn)):
        shutil.copy(os.path.join(A, fn), os.path.join(fig_dir, fn))
print("tablas.json y figuras listas en", out_src)
