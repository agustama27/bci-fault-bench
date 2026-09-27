"""Convierte las salidas de analyze.py en tablas para contenido.js (formato APA,
decimales con coma) y copia las figuras a la carpeta del manuscrito.

Uso: python scripts/resultados_tablas.py --analysis results/analysis [--analysis-eegnet results/analysis-eegnet]
        --manuscrito ../30-TFG/Entregas/Modulo-2
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
ap.add_argument("--analysis-eegnet", default=None)
ap.add_argument("--manuscrito", required=True)
a = ap.parse_args()
A = a.analysis
out_src = os.path.join(a.manuscrito, "src")
fig_dir = os.path.join(out_src, "fig")
os.makedirs(fig_dir, exist_ok=True)

ORDER = ["none", "loss", "jitter", "delay", "disconnect", "burst_trial", "disconnect_trial"]
KIND_ES = {"none": "Referencia", "loss": "Pérdida de muestras", "jitter": "*Jitter*", "delay": "Retraso",
           "disconnect": "Desconexión", "burst_trial": "Pérdida contigua en el ensayo", "disconnect_trial": "Desconexión en el ensayo"}
UNIT = {"loss": lambda s: f"{s*100:g} %", "jitter": lambda s: f"{s*1000:g} ms", "delay": lambda s: f"{s*1000:g} ms",
        "disconnect": lambda s: f"{s:g} s", "burst_trial": lambda s: f"{s*100:g} % de la ventana", "disconnect_trial": lambda s: f"{s:g} s"}


def f(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if d == 0:
        return f"{x:,.0f}".replace(",", ".")
    return f"{x:.{d}f}".replace(".", ",")


def sev(kind, s):
    return UNIT[kind](s).replace(".", ",") if kind in UNIT else "—"


def star(p):
    return "*" if (p is not None and not np.isnan(p) and p < 0.05) else ""


def kinds_present(df):
    return [k for k in ORDER[1:] if (df.kind == k).any()]


def load(A):
    return (pd.read_csv(os.path.join(A, "descriptivos.csv")), pd.read_csv(os.path.join(A, "t_desempeno.csv")),
            pd.read_csv(os.path.join(A, "t_infra.csv")))


def friedman_of(t):
    return {k: (None if np.isnan(g.friedman_chi2.iloc[0]) else float(g.friedman_chi2.iloc[0]),
                None if np.isnan(g.friedman_p.iloc[0]) else float(g.friedman_p.iloc[0]), int(g.n.iloc[0]))
            for k, g in t.groupby("kind")}


def lookup(t, kind, s):
    q = t[(t.kind == kind) & (np.isclose(t.severity, s))]
    return q.iloc[0] if len(q) else None


desc, t_b, t_i = load(A)
ref = desc[desc.kind == "none"].iloc[0]
has_net = a.analysis_eegnet is not None and os.path.exists(os.path.join(a.analysis_eegnet, "t_desempeno.csv"))
if has_net:
    desc_n, t_n, _ = load(a.analysis_eegnet)
    ref_n = desc_n[desc_n.kind == "none"].iloc[0]

tablas = {}

# --- Tabla desempeño (obj. 3): ambos decodificadores
head = ["Tipo de fallo", "Severidad", "CSP+LDA: *balanced accuracy*", "CSP+LDA: *p* (Holm)"]
if has_net:
    head += ["EEGNet: *balanced accuracy*", "EEGNet: *p* (Holm)"]
head += ["Decisiones válidas"]
rows = [["Referencia", "—", f(ref.bacc, 3), "—"] + ([f(ref_n.bacc, 3), "—"] if has_net else []) + [f(ref.valid_rate, 3)]]
for kind in kinds_present(t_b):
    for _, r in t_b[t_b.kind == kind].sort_values("severity").iterrows():
        d = lookup(desc, kind, r.severity)
        row = [KIND_ES[kind], sev(kind, r.severity), f(d.bacc, 3), f(r.p_holm, 3) + star(r.p_holm)]
        if has_net:
            rn, dn = lookup(t_n, kind, r.severity), lookup(desc_n, kind, r.severity)
            row += [f(dn.bacc, 3) if dn is not None else "—", (f(rn.p_holm, 3) + star(rn.p_holm)) if rn is not None else "—"]
        row += [f(d.valid_rate, 3)]
        rows.append(row)
tablas["desempeno"] = dict(headers=head, rows=rows, friedman=friedman_of(t_b), friedman_eegnet=(friedman_of(t_n) if has_net else {}))

# --- Tabla infraestructura (obj. 2)
rows = [["Referencia", "—", f(ref.recv_ratio, 3), f(ref.n_gaps, 0), f(ref.lat_mean_ms, 1), f(ref.ia_std_ms, 1), f(ref.secs_no_data, 0)]]
for kind in kinds_present(desc):
    for s in sorted(desc[desc.kind == kind].severity.unique()):
        d = lookup(desc, kind, s)

        def p_of(metric, kind=kind, s=s):
            q = t_i[(t_i.metric == metric) & (t_i.kind == kind) & (np.isclose(t_i.severity, s))]
            return star(float(q.p_holm.iloc[0])) if len(q) else ""
        rows.append([KIND_ES[kind], sev(kind, s), f(d.recv_ratio, 3) + p_of("recv_ratio"), f(d.n_gaps, 0) + p_of("n_gaps"),
                     f(d.lat_mean_ms, 1) + p_of("lat_mean_ms"), f(d.ia_std_ms, 1) + p_of("ia_std_ms"),
                     f(d.secs_no_data, 0) + p_of("secs_no_data")])
tablas["infra"] = dict(
    headers=["Tipo de fallo", "Severidad", "Muestras recibidas / esperadas", "Huecos", "Latencia media (ms)", "Desv. intervalo entre bloques (ms)", "Segundos sin datos"],
    rows=rows)

# --- Tabla divergencia (obj. 4)
t_d = pd.read_csv(os.path.join(A, "t_divergencia.csv"))
t_d = t_d.assign(_o=t_d.kind.map({k: i for i, k in enumerate(ORDER)})).sort_values(["_o", "severity"])
rows = [[KIND_ES[r.kind], sev(r.kind, r.severity) if r.kind != "none" else "—", f(r.silent, 3), f(r.loud, 3), f(r.divergent, 3)] for _, r in t_d.iterrows()]
tablas["divergencia"] = dict(headers=["Tipo de fallo", "Severidad", "Operativo y degradado", "No operativo y no degradado", "Total divergentes"], rows=rows)

# --- Tabla detectores (obj. 5)
t_det = pd.read_csv(os.path.join(A, "t_detectores.csv"))
NAMES = {"umbrales": "Umbrales", "logistic": "Regresión logística", "random_forest": "*Random forest*", "gradient_boosting": "*Gradient boosting*"}
rows = [[NAMES.get(r.detector, r.detector), f(r.precision, 3), f(r.recall, 3), f(r.f1, 3), f(r.fpr, 3), f(r.detection_delay_s, 1), f"{int(r.episodes_detected)}/{int(r.episodes)}"] for _, r in t_det.iterrows()]
tablas["detectores"] = dict(headers=["Detector", "Precisión", "Sensibilidad", "F1", "Tasa de falsos positivos", "Retardo de detección (s)", "Episodios detectados"], rows=rows)

tablas["umbrales"] = json.load(open(os.path.join(A, "thresholds.json")))

with open(os.path.join(out_src, "tablas.json"), "w", encoding="utf-8") as fh:
    json.dump(tablas, fh, ensure_ascii=False, indent=1)
for fn in ("fig_degradacion.png", "fig_latencia.png", "fig_integridad.png"):
    if os.path.exists(os.path.join(A, fn)):
        shutil.copy(os.path.join(A, fn), os.path.join(fig_dir, fn))
if has_net and os.path.exists(os.path.join(a.analysis_eegnet, "fig_degradacion.png")):
    shutil.copy(os.path.join(a.analysis_eegnet, "fig_degradacion.png"), os.path.join(fig_dir, "fig_degradacion_eegnet.png"))
print("tablas.json y figuras listas en", out_src)
