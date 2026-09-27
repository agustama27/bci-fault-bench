"""Extrae de la campaña y del análisis los números que cita el texto de Resultados
→ <manuscrito>/src/numeros.json (todo ya formateado en español).

Uso: python scripts/resultados_numeros.py --analysis results/analysis --campana results/vm/campana \
        --piloto results/vm/piloto --solo results/vm/piloto-solo --offline results/offline_bacc.csv \
        --manuscrito ../30-TFG/Entregas/Modulo-2 --horas 4.3
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bcibench import metrics as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--analysis", required=True)
ap.add_argument("--campana", required=True)
ap.add_argument("--piloto", required=True)
ap.add_argument("--solo", required=True)
ap.add_argument("--offline", required=True)
ap.add_argument("--manuscrito", required=True)
ap.add_argument("--horas", type=float, required=True)
a = ap.parse_args()
A = a.analysis


def f(x, d=3):
    if d == 0:
        return f"{x:,.0f}".replace(",", ".")
    return f"{x:.{d}f}".replace(".", ",")


def lst(vals, d=3, sep=", ", last=" y "):
    s = [f(v, d) for v in vals]
    return sep.join(s[:-1]) + last + s[-1] if len(s) > 1 else s[0]


e = pd.read_csv(os.path.join(A, "execs.csv"))
desc = pd.read_csv(os.path.join(A, "descriptivos.csv")).set_index("label")
t_i = pd.read_csv(os.path.join(A, "t_infra.csv"))
t_b = pd.read_csv(os.path.join(A, "t_desempeno.csv"))
t_d = pd.read_csv(os.path.join(A, "t_divergencia.csv")).set_index("label")
t_det = pd.read_csv(os.path.join(A, "t_detectores.csv")).set_index("detector")
thr = json.load(open(os.path.join(A, "thresholds.json")))["thresholds"]
W = pd.read_csv(os.path.join(A, "windows.csv"))
off = pd.read_csv(a.offline).set_index("subject")
ref = e[e.kind == "none"]

N = {}
N["n_execs"] = int(len(e))
N["horas_campana"] = f(a.horas, 1)
N["timing_p99_us"] = f(ref.timing_p99_ms.median() * 1000, 1)
N["timing_p99_max_us"] = f(ref.timing_p99_ms.max() * 1000, 1)

# control de paralelismo
pil, _, _ = M.load_campaign(a.piloto)
solo, _, _ = M.load_campaign(a.solo)
pr = pil[pil.kind == "none"].set_index("subject")
so = solo.set_index("subject")
N["ctrl_lat_ms"] = f((pr.lat_mean_ms - so.lat_mean_ms).median(), 2)
N["ctrl_ia_ms"] = f((pr.ia_std_ms - so.ia_std_ms).median(), 2)

# tabla 2
on = ref.groupby("subject").bacc.median()
N["tabla2"] = [[str(s), f(off.loc[s, "runs01"]), f(on.loc[s]), f(thr[str(s)])] for s in sorted(on.index)]

# infra
def cond(kind, sev, col):
    return desc.loc[f"{kind}-{sev:g}" if kind != "loss" else f"loss-random-{sev:g}", col]

N["recv_loss"] = lst([cond("loss", s, "recv_ratio") for s in (0.01, 0.05, 0.1)])
N["gaps_loss"] = lst([cond("loss", s, "n_gaps") for s in (0.01, 0.05, 0.1)], 0)
N["lat_loss_delta"] = lst([cond("loss", s, "lat_mean_ms") - desc.loc["ref", "lat_mean_ms"] for s in (0.01, 0.05, 0.1)], 2)
N["ia_ref"] = f(desc.loc["ref", "ia_std_ms"], 1)
N["lat_ref"] = f(desc.loc["ref", "lat_mean_ms"], 1)
N["ia_jitter"] = lst([cond("jitter", s, "ia_std_ms") for s in (0.01, 0.05, 0.1)], 1)
N["lat_jitter"] = lst([cond("jitter", s, "lat_mean_ms") for s in (0.01, 0.05, 0.1)], 1)
N["lat_delay"] = lst([cond("delay", s, "lat_mean_ms") for s in (0.05, 0.1, 0.25)], 1)
sig = t_i[(t_i.kind.isin(["loss", "jitter", "delay"])) & (t_i.p_holm < 0.05)]
N["r_max"] = f(sig.r.min(), 2)
N["r_top"] = f(sig.r.max(), 2)
N["p_min"] = f(sig.p_holm.max(), 3)
N["recv_disc"] = lst([cond("disconnect", s, "recv_ratio") for s in (0.5, 1, 3)])
dur = e.groupby("label").secs_total.median()
N["recv_disc_nominal"] = lst([1 - 5 * s / float(dur.get(f"disconnect-{s:g}", 387)) for s in (0.5, 1, 3)])
ia_max_1 = cond("disconnect", 1, "ia_max_ms"); ia_max_3 = cond("disconnect", 3, "ia_max_ms")
N["ia_max_disc"] = f"{f(ia_max_1, 0)} y {f(ia_max_3, 0)}"
N["recon_ms"] = f(np.mean([ia_max_1 - 1000, ia_max_3 - 3000]), 0)
N["nodata_disc"] = lst([cond("disconnect", s, "secs_no_data") for s in (0.5, 1, 3)], 0)

# desempeño
N["bacc_ref"] = f(desc.loc["ref", "bacc"])
N["conf_rango"] = f"{f(desc.conf.min(), 2)} y {f(desc.conf.max(), 2)}"
N["dd_ref"] = f(desc.loc["ref", "decision_delay_ms"], 0)
N["dd_delay"] = lst([cond("delay", s, "decision_delay_ms") for s in (0.05, 0.1, 0.25)], 0)
N["dd_jitter"] = lst([cond("jitter", s, "decision_delay_ms") for s in (0.01, 0.05, 0.1)], 0)
others = [cond("loss", s, "decision_delay_ms") for s in (0.01, 0.05, 0.1)] + [cond("disconnect", s, "decision_delay_ms") for s in (0.5, 1, 3)]
N["dd_otros"] = f"entre {f(min(others), 0)} y {f(max(others), 0)}"

# divergencia
N["silent_max"] = f(t_d.silent.max())
N["loud_disc"] = lst([t_d.loc[f"disconnect-{s:g}", "loud"] for s in (0.5, 1, 3)])
V = W.dropna(subset=["degraded"])
N["n_windows"] = f"{int(len(V)):,}".replace(",", ".")
N["n_pos"] = f"{int(V.degraded.sum()):,}".replace(",", ".")
N["episodios"] = int(t_det.episodes.iloc[0]) if len(t_det) else 0

# detectores
if len(t_det):
    N["prec_max"] = f(np.ceil(t_det.precision.max() * 100) / 100, 2)
    N["thr_recall"] = f(t_det.loc["umbrales", "recall"]); N["thr_fpr"] = f(t_det.loc["umbrales", "fpr"])
    N["lr_recall"] = f(t_det.loc["logistic", "recall"]); N["lr_fpr"] = f(t_det.loc["logistic", "fpr"])
    N["rf_recall"] = f(t_det.loc["random_forest", "recall"]); N["gb_recall"] = f(t_det.loc["gradient_boosting", "recall"])

out = os.path.join(a.manuscrito, "src", "numeros.json")
json.dump(N, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("numeros.json ->", out)
for k, v in N.items():
    if k != "tabla2":
        print(f"  {k}: {v}")
