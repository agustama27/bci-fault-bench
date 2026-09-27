"""Análisis completo de una campaña → tablas (CSV + Markdown) y figuras (PNG).

Uso: python scripts/analyze.py --root results/campana --out results/analysis --window 8 --q 5

Salidas (en --out):
  execs.csv                  una fila por ejecución con todas las variables
  t_desempeno.csv/.md        Tabla: balanced accuracy por tipo y severidad (mediana entre sujetos) + pruebas
  t_infra.csv/.md            Tabla: integridad, temporalidad, disponibilidad por condición + pruebas
  t_divergencia.csv/.md      Tabla: proporción de ventanas divergentes por condición
  t_detectores.csv/.md       Tabla: umbrales vs modelos (precisión, sensibilidad, F1, retardo)
  fig_degradacion.png        Curvas de degradación: bacc vs severidad por tipo
  fig_infra.png              Variables de infraestructura vs severidad
  thresholds.json            umbral por sujeto, ventana, percentil
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bcibench import metrics as M, stats as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--window", type=int, default=8, help="ensayos por ventana móvil (W)")
ap.add_argument("--q", type=float, default=5.0, help="percentil (rule=percentile) o factor (rule=mad)")
ap.add_argument("--thr-rule", default="percentile", choices=["percentile", "min", "mad"])
ap.add_argument("--trials-file", default="trials.csv", help="trials.csv | trials_eegnet.csv | trials_csp_offline.csv")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

execs, trials, tele = M.load_campaign(a.root, a.trials_file)
if execs.empty:
    sys.exit("sin ejecuciones en " + a.root)
execs.to_csv(os.path.join(a.out, "execs.csv"), index=False)
print(f"{len(execs)} ejecuciones, {execs.subject.nunique()} sujetos, {execs.label.nunique()} condiciones")


def md(df: pd.DataFrame, path: str, floatfmt=3):
    with open(path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False, floatfmt=f".{floatfmt}f"))


# ---------------------------------------------------------------- 1. desempeño (obj. 3) e infraestructura (obj. 2)
FUNC = ["bacc", "acc", "conf", "valid_rate", "decision_delay_ms"]
INFRA = ["recv_ratio", "n_gaps", "gap_s", "lat_mean_ms", "lat_max_ms", "ia_std_ms", "ia_max_ms", "secs_no_data", "reconnections"]
med = M.subject_median(execs, FUNC + INFRA)
med.to_csv(os.path.join(a.out, "medianas_por_sujeto.csv"), index=False)

desc = (med.groupby(["kind", "severity", "label"])[FUNC + INFRA]
           .median().reset_index().sort_values(["kind", "severity"]))
desc["kind_es"] = desc["kind"].map(M.KIND_ES)
desc.to_csv(os.path.join(a.out, "descriptivos.csv"), index=False)

t_bacc = S.by_kind_tests(med, "bacc")
t_bacc["kind_es"] = t_bacc["kind"].map(M.KIND_ES)
t_bacc.to_csv(os.path.join(a.out, "t_desempeno.csv"), index=False)
md(t_bacc[["kind_es", "severity", "n", "median_ref", "median_cond", "diff", "friedman_chi2", "friedman_p", "p_holm", "r"]],
   os.path.join(a.out, "t_desempeno.md"))

rows = []
for m in ["recv_ratio", "n_gaps", "lat_mean_ms", "ia_std_ms", "secs_no_data", "valid_rate"]:
    t = S.by_kind_tests(med, m)
    t.insert(0, "metric", m)
    rows.append(t)
t_infra = pd.concat(rows, ignore_index=True)
t_infra["kind_es"] = t_infra["kind"].map(M.KIND_ES)
t_infra.to_csv(os.path.join(a.out, "t_infra.csv"), index=False)
md(t_infra[["metric", "kind_es", "severity", "n", "median_ref", "median_cond", "p_holm", "r"]], os.path.join(a.out, "t_infra.md"))

# ---------------------------------------------------------------- 2. ventana móvil, umbral por sujeto, divergencia (obj. 4)
rolls = {eid: M.rolling_bacc(tr, a.window) for eid, tr in trials.items()}
thr = {}
for s in sorted(execs.subject.unique()):
    refs = execs[(execs.subject == s) & (execs.kind == "none")].exec_id
    thr[int(s)] = M.reference_threshold([rolls[e] for e in refs], a.q, a.thr_rule)
json.dump(dict(window=a.window, q=a.q, rule=a.thr_rule, thresholds=thr),
          open(os.path.join(a.out, "thresholds.json"), "w"), indent=2)

windows = []
for _, e in execs.iterrows():
    w = M.window_table(tele[e.exec_id], rolls[e.exec_id], thr[int(e.subject)])
    w["exec_id"], w["subject"], w["kind"], w["severity"], w["label"] = e.exec_id, e.subject, e.kind, e.severity, e.label
    windows.append(w)
W = pd.concat(windows, ignore_index=True)
W.to_csv(os.path.join(a.out, "windows.csv"), index=False)

div = (W.groupby(["exec_id", "subject", "kind", "severity", "label"])
        .apply(lambda g: pd.Series(M.divergence(g)), include_groups=False).reset_index())
div_med = M.subject_median(div, ["silent", "loud", "divergent"])
t_div = (div_med.groupby(["kind", "severity", "label"])[["silent", "loud", "divergent"]]
                .median().reset_index().sort_values(["kind", "severity"]))
t_div["kind_es"] = t_div["kind"].map(M.KIND_ES)
t_div.to_csv(os.path.join(a.out, "t_divergencia.csv"), index=False)
md(t_div[["kind_es", "severity", "silent", "loud", "divergent"]], os.path.join(a.out, "t_divergencia.md"))

# ---------------------------------------------------------------- 3. detectores (obj. 5)
V = W.dropna(subset=["degraded"]).copy()
y = V["degraded"].to_numpy(dtype=int)
subj = V["subject"].to_numpy()
tsec = V["sec"].to_numpy(dtype=float)
det_rows = []
if len(V) and y.min() != y.max():
    # umbrales: límites de la referencia, LOSO también (límites de los otros 8 sujetos)
    alarm_thr = np.zeros(len(V), dtype=bool)
    for s in np.unique(subj):
        ref_other = V[(V.subject != s) & (V.kind == "none")]
        lim = S.reference_limits(ref_other)
        m = subj == s
        alarm_thr[m] = S.threshold_detector(V[m], lim)
    det_rows.append(dict(detector="umbrales", **S.detection_scores(y, alarm_thr, tsec)))
    preds = S.loso_models(V, subj, y)
    for name, p in preds.items():
        det_rows.append(dict(detector=name, **S.detection_scores(y, p.astype(bool), tsec)))
t_det = pd.DataFrame(det_rows)
t_det.to_csv(os.path.join(a.out, "t_detectores.csv"), index=False)
if len(t_det):
    md(t_det, os.path.join(a.out, "t_detectores.md"))

# ---------------------------------------------------------------- 4. figuras (APA: sin color, ejes rotulados)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
markers = ["o", "s", "^", "D"]
SEV_LABEL = {
    "loss": lambda s: f"{s*100:g} %".replace(".", ","),
    "jitter": lambda s: f"{s*1000:g} ms".replace(".", ","),
    "delay": lambda s: f"{s*1000:g} ms".replace(".", ","),
    "disconnect": lambda s: f"{s:g} s".replace(".", ","),
    "burst_trial": lambda s: f"{s*100:g} %",
    "disconnect_trial": lambda s: f"{s:g} s".replace(".", ","),
}


def fig_by_kind(metric: str, ylabel: str, fname: str):
    kinds = [k for k in M.KINDS if (med.kind == k).any()]
    ncols = min(4, len(kinds)); nrows = int(np.ceil(len(kinds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.0 * nrows), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(kinds):]:
        ax.axis("off")
    for ax, kind, mk in zip(axes, kinds, (markers * 2)):
        sub = med[med.kind == kind]
        ref = med[med.kind == "none"][metric].median()
        sevs = sorted(sub.severity.unique())
        xs = [0] + list(range(1, len(sevs) + 1))
        ys = [ref] + [sub[sub.severity == s][metric].median() for s in sevs]
        lo = [med[med.kind == "none"][metric].quantile(.25)] + [sub[sub.severity == s][metric].quantile(.25) for s in sevs]
        hi = [med[med.kind == "none"][metric].quantile(.75)] + [sub[sub.severity == s][metric].quantile(.75) for s in sevs]
        ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], fmt=f"{mk}-", color="k",
                    ecolor="0.5", capsize=3, lw=1)
        ax.set_xticks(xs)
        sev_label = SEV_LABEL.get(kind, lambda s: f"×{s:g}".replace(".", ","))
        ax.set_xticklabels(["ref."] + [sev_label(s) for s in sevs])
        ax.set_title(M.KIND_ES[kind].capitalize(), fontsize=10)
        ax.set_xlabel("Severidad")
        ax.tick_params(axis="x", labelsize=8)
    for r in range(nrows):
        axes[r * ncols].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, fname), dpi=300)
    plt.close(fig)


fig_by_kind("bacc", "Balanced accuracy", "fig_degradacion.png")
fig_by_kind("lat_mean_ms", "Latencia media (ms)", "fig_latencia.png")
fig_by_kind("recv_ratio", "Muestras recibidas / esperadas", "fig_integridad.png")
print("listo:", a.out)
