"""Tabla comparativa de decodificadores por condición (líneas futuras 4 y 5).

Lee <analysis-root>/analysis-<variante>/t_desempeno.csv y medianas_por_sujeto.csv
(salidas de analyze.py con --trials-file trials_<variante>.csv) y escribe en --out:
  tabla_variantes.csv/.md   condición × variante: mediana entre sujetos de la balanced
                            accuracy (cada sujeto = mediana de sus corridas), p_holm y r
                            contra la referencia de la PROPIA variante
  tabla_por_sujeto.csv/.md  sujeto × variante en las condiciones estructuradas (y referencia)
  tabla_limpia.csv/.md      desempeño sin fallo: ensayos limpios de las ejecuciones "ref"
                            (sesión 2), balanced accuracy agrupada por sujeto
Uso: python scripts/compare_variants.py --analysis-root results --eval-root results/campana2-eval \
        --out results/robustez --variants csp_online csp_offline eegnet ...
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bcibench.metrics import KIND_ES, KINDS  # noqa: E402

TRIALS = {"csp_online": "trials.csv", "csp_offline": "trials_csp_offline.csv"}
STRUCT = ["burst_trial", "disconnect_trial"]

ap = argparse.ArgumentParser()
ap.add_argument("--analysis-root", default="results")
ap.add_argument("--eval-root", default="results/campana2-eval")
ap.add_argument("--out", default="results/robustez")
ap.add_argument("--variants", nargs="+", required=True)
ap.add_argument("--tag", default="", help="sufijo de los archivos de salida")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def fmt_p(p):
    return "—" if pd.isna(p) else (f"{p:.3f}".lstrip("0") if p >= 0.001 else "<.001")


long_rows, subj_rows, n_subj = [], [], {}
for v in a.variants:
    d = os.path.join(a.analysis_root, f"analysis-{v}")
    t = pd.read_csv(os.path.join(d, "t_desempeno.csv"))
    med = pd.read_csv(os.path.join(d, "medianas_por_sujeto.csv"))
    n_subj[v] = int(med.subject.nunique())
    ref = med[med.kind == "none"]
    long_rows.append(dict(kind="none", severity=0.0, variant=v, n=len(ref), median=ref.bacc.median(),
                          p_holm=np.nan, r=np.nan, diff=0.0))
    for _, r in t.iterrows():
        long_rows.append(dict(kind=r.kind, severity=r.severity, variant=v, n=int(r.n), median=r.median_cond,
                              p_holm=r.p_holm, r=r.r, diff=r["diff"]))
    for _, r in med[med.kind.isin(["none"] + STRUCT)].iterrows():
        subj_rows.append(dict(subject=int(r.subject), kind=r.kind, severity=r.severity, variant=v, bacc=r.bacc))

L = pd.DataFrame(long_rows)
order = {k: i for i, k in enumerate(["none"] + KINDS)}
L["o"] = L.kind.map(order)
L = L.sort_values(["o", "severity"]).drop(columns="o")
L.to_csv(os.path.join(a.out, f"tabla_variantes{a.tag}.csv"), index=False)


def cond_name(k, s):
    return "referencia" if k == "none" else f"{KIND_ES[k]} {s:g}"


L["condicion"] = [cond_name(k, s) for k, s in zip(L.kind, L.severity)]
L["cell"] = [f"{m:.3f}" if k == "none" else f"{m:.3f} ({fmt_p(p)}; {'—' if pd.isna(r) else f'{r:.2f}'})"
             for k, m, p, r in zip(L.kind, L["median"], L.p_holm, L.r)]
W = L.pivot_table(index=["condicion"], columns="variant", values="cell", aggfunc="first", sort=False)
W = W[[v for v in a.variants]]
W.columns = [f"{v} (n={n_subj[v]})" for v in W.columns]
W = W.reset_index()
with open(os.path.join(a.out, f"tabla_variantes{a.tag}.md"), "w", encoding="utf-8") as f:
    f.write("Celda: mediana entre sujetos de la balanced accuracy (p_holm; r) vs. la referencia de la propia variante. "
            "n = sujetos cubiertos.\n\n")
    f.write(W.to_markdown(index=False))

S = pd.DataFrame(subj_rows)
S["condicion"] = [cond_name(k, s) for k, s in zip(S.kind, S.severity)]
S["o"] = S.kind.map(order)
PS = S.sort_values(["o", "severity", "subject"]).pivot_table(index=["condicion", "subject"], columns="variant",
                                                             values="bacc", sort=False)
PS = PS[[v for v in a.variants]].reset_index()
PS.to_csv(os.path.join(a.out, f"tabla_por_sujeto{a.tag}.csv"), index=False)
with open(os.path.join(a.out, f"tabla_por_sujeto{a.tag}.md"), "w", encoding="utf-8") as f:
    f.write(PS.to_markdown(index=False, floatfmt=".3f"))

# ---------------------------------------------------------------- desempeño limpio (ejecuciones ref)
clean = []
for v in a.variants:
    fn = TRIALS.get(v, f"trials_{v}.csv")
    per = {}
    for d in sorted(glob.glob(os.path.join(a.eval_root, "s*-ref"))):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            continue
        tr = pd.read_csv(p)
        tr = tr[tr.valid == 1]
        per.setdefault(int(os.path.basename(d)[1:3]), []).append(tr)
    for s, trs in per.items():
        tr = pd.concat(trs)
        clean.append(dict(variant=v, subject=s, n_trials=len(tr),
                          bacc=balanced_accuracy_score(tr.label, tr.pred.astype(float).astype(int))))
C = pd.DataFrame(clean)
C.to_csv(os.path.join(a.out, f"tabla_limpia{a.tag}.csv"), index=False)
CP = C.pivot(index="subject", columns="variant", values="bacc")[[v for v in a.variants if v in set(C.variant)]]
CP.loc["mediana"] = CP.median()
with open(os.path.join(a.out, f"tabla_limpia{a.tag}.md"), "w", encoding="utf-8") as f:
    f.write("Balanced accuracy sin fallo: ensayos válidos de las 5 corridas de referencia por sujeto (sesión 2), agrupados.\n\n")
    f.write(CP.reset_index().to_markdown(index=False, floatfmt=".3f"))
print(W.to_string())
