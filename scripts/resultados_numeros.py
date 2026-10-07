"""Extrae de la campaña y de los análisis los números que cita el texto de Resultados
→ <manuscrito>/src/numeros.json (formateados en español).

Uso: python scripts/resultados_numeros.py --analysis results/analysis --analysis-eegnet results/analysis-eegnet
        --analysis-csp-offline results/analysis-csp-offline --campana results/vm/campana2
        --piloto results/vm/piloto --solo results/vm/piloto-solo --offline results/offline_bacc.csv
        --eegnet-offline results/eegnet_offline.csv --manuscrito ../30-TFG/Entregas/Modulo-2 --horas 8.3
"""
import argparse
import glob
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
ap.add_argument("--analysis-eegnet", default=None)
ap.add_argument("--analysis-csp-offline", default=None)
ap.add_argument("--campana", required=True)
ap.add_argument("--piloto", required=True)
ap.add_argument("--solo", required=True)
ap.add_argument("--offline", required=True)
ap.add_argument("--eegnet-offline", default=None)
ap.add_argument("--manuscrito", required=True)
ap.add_argument("--horas", type=float, required=True)
a = ap.parse_args()
A = a.analysis


def f(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if d == 0:
        return f"{x:,.0f}".replace(",", ".")
    return f"{x:.{d}f}".replace(".", ",")


def lst(vals, d=3, sep=", ", last=" y "):
    s = [f(v, d) for v in vals]
    return sep.join(s[:-1]) + last + s[-1] if len(s) > 1 else s[0]


SEV = {"loss": (0.01, 0.05, 0.1), "jitter": (0.01, 0.05, 0.1), "delay": (0.05, 0.1, 0.25),
       "disconnect": (0.5, 1, 3), "burst_trial": (0.1, 0.25, 0.4), "disconnect_trial": (0.5, 1, 2)}
LABEL = {"loss": "loss-random-{s:g}", "jitter": "jitter-{s:g}", "delay": "delay-{s:g}", "disconnect": "disconnect-{s:g}",
         "burst_trial": "burst_trial-{s:g}", "disconnect_trial": "disconnect_trial-{s:g}"}
NAME = {"loss": "la pérdida del {p:g} %", "jitter": "el jitter de {p:g} ms", "delay": "el retraso de {p:g} ms",
        "disconnect": "la desconexión de {p:g} s", "burst_trial": "la pérdida contigua del {p:g} % de la ventana",
        "disconnect_trial": "la desconexión en el ensayo de {p:g} s"}


def human(kind, s):
    v = s * 100 if kind in ("loss", "burst_trial") else (s * 1000 if kind in ("jitter", "delay") else s)
    return NAME[kind].format(p=v).replace(".", ",")


def load(A):
    return (pd.read_csv(os.path.join(A, "execs.csv")), pd.read_csv(os.path.join(A, "descriptivos.csv")).set_index("label"),
            pd.read_csv(os.path.join(A, "t_infra.csv")), pd.read_csv(os.path.join(A, "t_desempeno.csv")))


e, desc, t_i, t_b = load(A)
t_d = pd.read_csv(os.path.join(A, "t_divergencia.csv")).set_index("label")
_ap = os.path.join(A, "t_detectores_ap.csv")
t_det = pd.read_csv(_ap if os.path.exists(_ap) else os.path.join(A, "t_detectores.csv")).set_index("detector")
thr = json.load(open(os.path.join(A, "thresholds.json")))["thresholds"]
W = pd.read_csv(os.path.join(A, "windows.csv"))
off = pd.read_csv(a.offline).set_index("subject")
ref = e[e.kind == "none"]
has_net = bool(a.analysis_eegnet) and os.path.exists(os.path.join(a.analysis_eegnet, "execs.csv"))
if has_net:
    e_n, desc_n, _, t_n = load(a.analysis_eegnet)
    off_n = pd.read_csv(a.eegnet_offline).set_index("subject") if a.eegnet_offline and os.path.exists(a.eegnet_offline) else None


def cond(kind, s, col, d=None):
    d = desc if d is None else d
    lab = LABEL[kind].format(s=s)
    return d.loc[lab, col] if lab in d.index else np.nan


def pval(kind, s, t=None):
    t = t_b if t is None else t
    q = t[(t.kind == kind) & (np.isclose(t.severity, s))]
    return float(q.p_holm.iloc[0]) if len(q) else np.nan


N = {}
N["n_execs"] = int(len(e))
N["n_execs_txt"] = f(len(e), 0)
N["horas_campana"] = f(a.horas, 1)
N["timing_p99_us"] = f(ref.timing_p99_ms.median() * 1000, 1)
N["timing_p99_max_us"] = f(ref.timing_p99_ms.max() * 1000, 1)
N["n_runs"] = int(e.run.astype(str).nunique())

# control de paralelismo: referencia de la campaña (en paralelo) vs sola (piloto-solo)
solo, _, _ = M.load_campaign(a.solo)
so = solo.set_index("subject")
par = ref[ref.subject.isin(so.index)].groupby("subject")[["lat_mean_ms", "ia_std_ms"]].median()
N["ctrl_lat_ms"] = f((par.lat_mean_ms - so.lat_mean_ms).median(), 2)
N["ctrl_ia_ms"] = f((par.ia_std_ms - so.ia_std_ms).median(), 2)

# tabla 2: por sujeto
on = ref.groupby("subject").bacc.median()
_col = "runs04_median" if "runs04_median" in off.columns else ("all" if "all" in off.columns else "runs01")
_ncol = "eegnet_runs04_median" if (has_net and off_n is not None and "eegnet_runs04_median" in off_n.columns) else "eegnet_all"
rows = []
on_n = e_n[e_n.kind == "none"].groupby("subject").bacc.median() if has_net else None
for s in sorted(on.index):
    r = [str(s), f(off.loc[s, _col]), f(on.loc[s])]
    if has_net:
        r += [f(off_n.loc[s, _ncol]) if off_n is not None and s in off_n.index else "—", f(on_n.loc[s]) if s in on_n.index else "—"]
    r += [f(thr[str(s)])]
    rows.append(r)
N["tabla2"] = rows
N["tabla2_has_net"] = bool(has_net)
N["tabla2_iguales"] = int(sum(abs(off.loc[s, _col] - on.loc[s]) < 0.005 for s in on.index))
N["tabla2_distintos"] = ", ".join(str(s) for s in sorted(on.index) if abs(off.loc[s, _col] - on.loc[s]) >= 0.005)
N["tabla2_maxdiff"] = f(max(abs(off.loc[s, _col] - on.loc[s]) for s in on.index))

# equivalencia en línea / fuera de línea (CSP): coincidencia de predicciones
agree = tot = 0
for d in glob.glob(os.path.join(a.campana, "s0*/")):
    p1, p2 = os.path.join(d, "trials.csv"), os.path.join(d, "trials_csp_offline.csv")
    if os.path.exists(p1) and os.path.exists(p2):
        t1, t2 = pd.read_csv(p1), pd.read_csv(p2)
        n = min(len(t1), len(t2))
        v = (t1.valid.values[:n] == 1) & (t2.valid.values[:n] == 1)
        agree += int((t1.pred.values[:n][v].astype(int) == t2.pred.values[:n][v].astype(int)).sum()); tot += int(v.sum())
N["equiv_pct"] = f(100 * agree / tot, 1) if tot else "—"
N["equiv_n"] = f(tot, 0)

# infraestructura, familia uniforme
N["recv_loss"] = lst([cond("loss", s, "recv_ratio") for s in SEV["loss"]])
N["gaps_loss"] = lst([cond("loss", s, "n_gaps") for s in SEV["loss"]], 0)
N["lat_loss_delta"] = lst([cond("loss", s, "lat_mean_ms") - desc.loc["ref", "lat_mean_ms"] for s in SEV["loss"]], 2)
N["ia_ref"] = f(desc.loc["ref", "ia_std_ms"], 1)
N["lat_ref"] = f(desc.loc["ref", "lat_mean_ms"], 1)
N["ia_jitter"] = lst([cond("jitter", s, "ia_std_ms") for s in SEV["jitter"]], 1)
N["lat_jitter"] = lst([cond("jitter", s, "lat_mean_ms") for s in SEV["jitter"]], 1)
N["lat_delay"] = lst([cond("delay", s, "lat_mean_ms") for s in SEV["delay"]], 1)
sig = t_i[(t_i.kind.isin(["loss", "jitter", "delay"])) & (t_i.p_holm < 0.05)]
N["r_max"] = f(sig.r.min(), 2); N["r_top"] = f(sig.r.max(), 2); N["p_min"] = f(sig.p_holm.max(), 3)
N["recv_disc"] = lst([cond("disconnect", s, "recv_ratio") for s in SEV["disconnect"]])
dur = e.groupby("label").secs_total.median()
N["recv_disc_nominal"] = lst([1 - 5 * s / float(dur.get(f"disconnect-{s:g}", 387)) for s in SEV["disconnect"]])
ia_max_1, ia_max_3 = cond("disconnect", 1, "ia_max_ms"), cond("disconnect", 3, "ia_max_ms")
N["ia_max_disc"] = f"{f(ia_max_1, 0)} y {f(ia_max_3, 0)}"
N["recon_ms"] = f(np.mean([ia_max_1 - 1000, ia_max_3 - 3000]), 0)
ia_max_05 = cond("disconnect", 0.5, "ia_max_ms")
N["ia_max_disc3"] = lst([ia_max_05, ia_max_1, ia_max_3], 0)
N["recon_excess"] = lst([ia_max_05 - 500, ia_max_1 - 1000, ia_max_3 - 3000], 0)
N["nodata_disc"] = lst([cond("disconnect", s, "secs_no_data") for s in SEV["disconnect"]], 0)

# infraestructura, familia estructurada
has_struct = LABEL["burst_trial"].format(s=0.25) in desc.index
N["has_struct"] = bool(has_struct)
if has_struct:
    N["recv_burst"] = lst([cond("burst_trial", s, "recv_ratio") for s in SEV["burst_trial"]])
    N["gaps_burst"] = lst([cond("burst_trial", s, "n_gaps") for s in SEV["burst_trial"]], 0)
    N["recv_disct"] = lst([cond("disconnect_trial", s, "recv_ratio") for s in SEV["disconnect_trial"]])
    N["nodata_disct"] = lst([cond("disconnect_trial", s, "secs_no_data") for s in SEV["disconnect_trial"]], 0)
    N["ia_max_disct"] = lst([cond("disconnect_trial", s, "ia_max_ms") for s in SEV["disconnect_trial"]], 0)
    N["valid_disct"] = lst([cond("disconnect_trial", s, "valid_rate") for s in SEV["disconnect_trial"]])
    N["valid_burst"] = lst([cond("burst_trial", s, "valid_rate") for s in SEV["burst_trial"]])
    ns = []
    for s in SEV["disconnect_trial"]:
        vals = []
        for d in glob.glob(os.path.join(a.campana, f"s0*-disconnect_trial-{s:g}/")):
            p = os.path.join(d, "trials.csv")
            if os.path.exists(p):
                vals.append(pd.read_csv(p).n_samples.median())
        ns.append(np.median(vals) if vals else np.nan)
    N["nsamp_disct"] = lst(ns, 0)

# desempeño: CSP en línea
N["bacc_ref"] = f(desc.loc["ref", "bacc"])
_u = desc[desc.index.str.match(r"^(loss|jitter|delay|disconnect)-")]
_diff = _u[abs(_u.bacc - desc.loc["ref", "bacc"]) >= 0.0005]
N["bacc_n_igual"] = int(len(_u) - len(_diff)); N["bacc_n_total"] = int(len(_u))
N["bacc_excepciones"] = "; ".join(f"{f(r.bacc)} en {human(r.kind, r.severity)}" for _, r in _diff.iterrows())
N["conf_rango"] = f"{f(desc.conf.min(), 2)} y {f(desc.conf.max(), 2)}"
N["dd_ref"] = f(desc.loc["ref", "decision_delay_ms"], 0)
N["dd_delay"] = lst([cond("delay", s, "decision_delay_ms") for s in SEV["delay"]], 0)
N["dd_jitter"] = lst([cond("jitter", s, "decision_delay_ms") for s in SEV["jitter"]], 0)
others = [cond("loss", s, "decision_delay_ms") for s in SEV["loss"]] + [cond("disconnect", s, "decision_delay_ms") for s in SEV["disconnect"]]
N["dd_otros"] = f"entre {f(min(others), 0)} y {f(max(others), 0)}"
N["sig_uniform_csp"] = [human(k, s) for k in ("loss", "jitter", "delay", "disconnect") for s in SEV[k] if pval(k, s) < 0.05]
if has_struct:
    N["sig_struct_csp"] = [human(k, s) for k in ("burst_trial", "disconnect_trial") for s in SEV[k] if pval(k, s) < 0.05]
    N["bacc_burst_csp"] = lst([cond("burst_trial", s, "bacc") for s in SEV["burst_trial"]])
    N["bacc_disct_csp"] = lst([cond("disconnect_trial", s, "bacc") for s in SEV["disconnect_trial"]])
    N["p_burst_csp"] = lst([pval("burst_trial", s) for s in SEV["burst_trial"]])
    N["p_disct_csp"] = lst([pval("disconnect_trial", s) for s in SEV["disconnect_trial"]])
if has_net:
    N["bacc_ref_net"] = f(desc_n.loc["ref", "bacc"])
    _un = [LABEL[k].format(s=s) for k in ("loss", "jitter", "delay", "disconnect") for s in SEV[k]]
    _vals = [desc_n.loc[l, "bacc"] for l in _un if l in desc_n.index]
    N["bacc_uniform_net_rango"] = f"{f(min(_vals))} y {f(max(_vals))}" if _vals else "—"
    N["sig_uniform_net"] = [human(k, s) for k in ("loss", "jitter", "delay", "disconnect") for s in SEV[k] if pval(k, s, t_n) < 0.05]
    if has_struct:
        N["sig_struct_net"] = [human(k, s) for k in ("burst_trial", "disconnect_trial") for s in SEV[k] if pval(k, s, t_n) < 0.05]
        N["bacc_burst_net"] = lst([cond("burst_trial", s, "bacc", desc_n) for s in SEV["burst_trial"]])
        N["bacc_disct_net"] = lst([cond("disconnect_trial", s, "bacc", desc_n) for s in SEV["disconnect_trial"]])
        N["p_burst_net"] = lst([pval("burst_trial", s, t_n) for s in SEV["burst_trial"]])
        N["p_disct_net"] = lst([pval("disconnect_trial", s, t_n) for s in SEV["disconnect_trial"]])

# divergencia y detectores
N["silent_max"] = f(t_d.silent.max())
N["loud_disc"] = lst([t_d.loc[f"disconnect-{s:g}", "loud"] for s in SEV["disconnect"]])
if has_struct:
    N["loud_disct"] = lst([t_d.loc[f"disconnect_trial-{s:g}", "loud"] if f"disconnect_trial-{s:g}" in t_d.index else np.nan for s in SEV["disconnect_trial"]])
    N["silent_burst"] = lst([t_d.loc[f"burst_trial-{s:g}", "silent"] if f"burst_trial-{s:g}" in t_d.index else np.nan for s in SEV["burst_trial"]])
    N["silent_disct"] = lst([t_d.loc[f"disconnect_trial-{s:g}", "silent"] if f"disconnect_trial-{s:g}" in t_d.index else np.nan for s in SEV["disconnect_trial"]])
V = W.dropna(subset=["degraded"])
N["n_windows"] = f(len(V), 0)
N["n_pos"] = f(int(V.degraded.sum()), 0)
N["episodios"] = int(t_det.episodes.iloc[0]) if len(t_det) else 0
if len(t_det):
    N["prec_max"] = f(np.ceil(t_det.precision.max() * 100) / 100, 2)
    for k, name in (("umbrales", "thr"), ("logistic", "lr"), ("random_forest", "rf"), ("gradient_boosting", "gb")):
        if k in t_det.index:
            N[f"{name}_recall"] = f(t_det.loc[k, "recall"]); N[f"{name}_fpr"] = f(t_det.loc[k, "fpr"])
            N[f"{name}_prec"] = f(t_det.loc[k, "precision"]); N[f"{name}_delay"] = f(t_det.loc[k, "detection_delay_s"], 1)
            N[f"{name}_f1"] = f(t_det.loc[k, "f1"])
            if "average_precision" in t_det.columns:
                N[f"{name}_ap"] = f(t_det.loc[k, "average_precision"]); N[f"{name}_lift"] = f(t_det.loc[k, "lift"], 1)
    if "prevalence" in t_det.columns:
        N["prevalencia"] = f(t_det.prevalence.iloc[0], 4)
    best = t_det.f1.idxmax()
    N["best_detector"] = {"umbrales": "el detector por umbrales", "logistic": "la regresión logística", "random_forest": "*random forest*", "gradient_boosting": "*gradient boosting*"}[best]
    N["best_f1"] = f(t_det.loc[best, "f1"])

out = os.path.join(a.manuscrito, "src", "numeros.json")
json.dump(N, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("numeros.json ->", out)
for k, v in N.items():
    if k != "tabla2":
        print(f"  {k}: {v}")
