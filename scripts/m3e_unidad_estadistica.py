"""M3e: ¿por qué el sujeto es la unidad estadística? Verificación empírica (2026-10-09).

Sobre la campaña definitiva (campaign-2026-09-27) calcula:
  1. ICC(1) de la balanced accuracy de referencia por sujeto (9 sujetos x 5 corridas) y su efecto de diseño.
  2. ICC(1) del acierto por ensayo (binario) por sujeto (1.080 ensayos de referencia).
  3. Para cada condición (CSP+LDA): la misma comparación contra la referencia con tres unidades
     (sujeto n=9, corrida n=45, ensayo n=1.080 con McNemar) y con un modelo mixto (intercepto aleatorio
     por sujeto) como control; además, la ICC de las diferencias Δ por sujeto.
Salida: campaign-2026-09-27/analysis-m3/m3e_unidad.csv y m3e_unidad_resumen.txt.
Requiere statsmodels para el modelo mixto (si falta, esa columna queda vacía).
Uso: python scripts/m3e_unidad_estadistica.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "campaign-2026-09-27"
OUT = A / "analysis-m3"
OUT.mkdir(exist_ok=True)


def icc1(groups):
    """ICC(1) de un factor, estimador ANOVA (Shrout & Fleiss, caso 1)."""
    groups = [np.asarray(g, float) for g in groups if len(g) > 0]
    k = len(groups)
    n_tot = sum(len(g) for g in groups)
    m = n_tot / k
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n_tot - k)
    den = msb + (m - 1) * msw
    return (msb - msw) / den if den > 0 else float("nan"), msb, msw, m


ex = pd.read_csv(A / "analysis" / "execs.csv")
ref = ex[ex.kind == "none"]
refx = ref[["subject", "run", "bacc"]].rename(columns={"bacc": "bacc_ref"})
lines = []

# 1. niveles: referencia
icc, msb, msw, m = icc1([g.bacc.values for _, g in ref.groupby("subject")])
F, pF = stats.f_oneway(*[g.bacc.values for _, g in ref.groupby("subject")])
de = 1 + (m - 1) * icc
lines.append(f"[1] Referencia, bacc por sujeto: ICC(1) = {icc:.3f}; F({ref.subject.nunique()-1},{len(ref)-ref.subject.nunique()}) = {F:.2f}, p = {pF:.2e}; "
             f"efecto de diseño (m={m:.0f}) = {de:.2f}; n efectivo = {len(ref)/de:.1f} de {len(ref)} corridas")
icc_all, *_ = icc1([g.bacc.values for _, g in ex.groupby("subject")])
lines.append(f"[1b] Todas las condiciones, bacc por sujeto: ICC(1) = {icc_all:.3f}")

# 2. ensayos binarios
tr = pd.read_csv(A / "analysis-m3b" / "m3b_A_trials.csv", encoding="latin-1")
reft = tr[tr.label_cond == "burst_trial-0.1"][["subject", "run", "i", "ref_ok_csp"]].drop_duplicates()
icc_t, _, _, mt = icc1([g.ref_ok_csp.values for _, g in reft.groupby("subject")])
de_t = 1 + (mt - 1) * icc_t
lines.append(f"[2] Ensayos de referencia (acierto binario) por sujeto: ICC(1) = {icc_t:.3f}; m = {mt:.0f}; efecto de diseño = {de_t:.1f}; "
             f"n efectivo = {len(reft)/de_t:.0f} de {len(reft)} ensayos")

# 3. tres unidades + modelo mixto por condición
try:
    import statsmodels.formula.api as smf
    HAVE_SM = True
except ImportError:
    HAVE_SM = False
rows = []
for lab, grp in ex[ex.label != "ref"].groupby("label"):
    d = grp.merge(refx, on=["subject", "run"])
    d["delta"] = d.bacc - d.bacc_ref
    ds = d.groupby("subject").delta.median()
    p_s = stats.wilcoxon(ds, zero_method="wilcox").pvalue if (ds != 0).any() else 1.0
    p_r = stats.wilcoxon(d.delta, zero_method="wilcox").pvalue if (d.delta != 0).any() else 1.0
    t = tr[tr.label_cond == lab]
    b = int(((t.ref_ok_csp == 1) & (t.ok_csp == 0)).sum())
    c = int(((t.ref_ok_csp == 0) & (t.ok_csp == 1)).sum())
    p_t = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    icc_d = icc1([g.delta.values for _, g in d.groupby("subject")])[0] if (d.delta != 0).any() else float("nan")
    p_m = est = float("nan")
    if HAVE_SM and (d.delta != 0).any():
        try:
            mdl = smf.mixedlm("delta ~ 1", d, groups=d["subject"]).fit(reml=True)
            p_m, est = mdl.pvalues["Intercept"], mdl.params["Intercept"]
        except Exception:
            pass
    elif not (d.delta != 0).any():
        p_m, est = 1.0, 0.0
    rows.append(dict(condicion=lab, delta_mediana_sujeto=ds.median(), icc_delta=icc_d, delta_mixto=est, p_mixto=p_m,
                     p_sujeto_n9=p_s, p_corrida_n45=p_r, p_ensayo_mcnemar_n1080=p_t, b_acertaba_falla=b, c_fallaba_acierta=c))
t = pd.DataFrame(rows)
t.to_csv(OUT / "m3e_unidad.csv", index=False)
sig = lambda col: int((t[col] < 0.05).sum())
lines.append(f"[3] Condiciones con p < 0,05 (de 18): sujeto {sig('p_sujeto_n9')} | corrida {sig('p_corrida_n45')} | "
             f"ensayo {sig('p_ensayo_mcnemar_n1080')} | modelo mixto {sig('p_mixto') if HAVE_SM else 'n/d'}")
lines.append(t.round(4).to_string(index=False))
(OUT / "m3e_unidad_resumen.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
