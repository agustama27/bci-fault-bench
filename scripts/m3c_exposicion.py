"""M3c · Tarea 2: barrido de exposición (K = 2/5/10/20 cortes de 1 s por corrida; campaña exploratoria post hoc).

Por ejecución: proporción de ensayos "tocados" (n_samples < 1000 o inválidos). Balanced accuracy CSP+LDA (en línea,
trials.csv) y EEGNet (fuera de línea, 'ceros' = EEGNet estándar, trials_eegnet_ceros.csv). Unidad estadística = sujeto
(mediana de sus 5 corridas); Friedman sobre {ref, K=2, 5, 10, 20} + Wilcoxon apareado de cada K vs ref con Holm dentro del
tipo y r = Z/sqrt(n) (bcibench.stats.by_kind_tests). Emparejamiento por ensayo con la referencia (mismo sujeto, corrida,
índice y onset): tasa de cambio de decisión en tocados vs no tocados.

Salidas (campaign-m3/analysis/): m3c_exp_exec.csv, m3c_exp_cond.csv, m3c_exp_tests.csv, m3c_exp_cambio.csv,
m3c_exp_trials.csv, m3c_exp_dosis.csv, m3c_exp_curva.png
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats as sps

from m3c_common import (NEW, NET_NEW, OUT, SWEEP, SWEEP_K, lab_filter, load, paired_trials, subject_med,
                        tests_group, wilson)

T_WIN = 387.0


def main():
    keep = lab_filter(SWEEP)
    ex_c, tr_c, _ = load(NEW, "trials.csv", keep)
    ex_n, _, _ = load(NEW, NET_NEW, keep)
    ex = ex_c.merge(ex_n[["exec_id", "bacc"]].rename(columns={"bacc": "bacc_net"}), on="exec_id")
    ex["K"] = ex.label.map(SWEEP_K)
    assert len(ex) == 225 and ex.groupby("label").size().eq(45).all(), ex.groupby("label").size()
    ex["n_outages_log"] = ex.outages
    # ------------- exposición por ejecución
    p_t, p_500, lostf = [], [], []
    for _, e in ex.iterrows():
        tr = tr_c[e.exec_id]
        n = tr.n_samples.to_numpy()
        touched = (n < 1000) | (tr.valid.to_numpy() == 0)
        p_t.append(float(touched.mean())); p_500.append(float((n < 500).mean())); lostf.append(float((1 - n / 1000).mean()))
    ex["p_touched"], ex["p_lt500"], ex["lost_frac"] = p_t, p_500, lostf
    ex["k_touched"] = (ex.p_touched * ex.n_trials).round().astype(int)
    ex["k_invalid"] = ex.n_trials - ex.n_valid
    ex.to_csv(os.path.join(OUT, "m3c_exp_exec.csv"), index=False)

    # ------------- pruebas (sujeto = mediana de corridas): kind 'sweep', severidad = K
    cols = ["bacc", "bacc_net", "p_touched", "lost_frac"]
    med = subject_med(ex.assign(kind=np.where(ex.label == "ref", "none", "sweep"), severity=ex.K.astype(float)), cols)
    med.to_csv(os.path.join(OUT, "m3c_exp_subject.csv"), index=False)
    tests = []
    for m, name in (("bacc", "CSP+LDA"), ("bacc_net", "EEGNet")):
        t = tests_group(med, m, "sweep")
        t.insert(0, "decoder", name)
        tests.append(t)
    T = pd.concat(tests, ignore_index=True).rename(columns={"severity": "K"})
    T.to_csv(os.path.join(OUT, "m3c_exp_tests.csv"), index=False)

    # ------------- tabla por condición
    rows = []
    for lab in SWEEP:
        g, gm = ex[ex.label == lab], med[med.label == lab] if lab != "ref" else med[med.kind == "none"]
        r = dict(label=lab, K=SWEEP_K[lab], n_exec=len(g), n_subj=len(gm),
                 p_touched_med_subj=gm.p_touched.median(), p_touched_q1=gm.p_touched.quantile(.25),
                 p_touched_q3=gm.p_touched.quantile(.75), p_touched_glob=g.k_touched.sum() / g.n_trials.sum(),
                 k_touched_total=int(g.k_touched.sum()), n_trials_total=int(g.n_trials.sum()),
                 k_invalid_total=int(g.k_invalid.sum()), lost_frac_glob=g.lost_frac.mean(),
                 outages_per_run_med=g.outages.median(),
                 bacc_csp_med=gm.bacc.median(), bacc_net_med=gm.bacc_net.median())
        rows.append(r)
    C = pd.DataFrame(rows)
    ref_s = med[med.kind == "none"].set_index("subject")
    for dec, col in (("csp", "bacc"), ("net", "bacc_net")):
        for i, lab in enumerate(SWEEP):
            if lab == "ref":
                continue
            s = med[med.label == lab].set_index("subject")
            d = (s[col] - ref_s[col]).dropna()
            C.loc[i, f"dBA_med_{dec}"] = d.median()
            C.loc[i, f"dBA_mean_{dec}"] = d.mean()
            C.loc[i, f"dBA_min_{dec}"] = d.min()
            C.loc[i, f"dBA_max_{dec}"] = d.max()
            C.loc[i, f"n_subj_peor_{dec}"] = int((d < 0).sum())
            C.loc[i, f"n_subj_mejor_{dec}"] = int((d > 0).sum())
    # modelo de ocupación previo: cada corte de 1 s deja tocados los ensayos cuya ventana de 4 s (+ el hueco de 1,52 s)
    # solapa -> E = K * (hueco + W) / T  (sin solapes ni zona de guarda)
    C["E_modelo"] = C.K * (1.52 + 4.0) / T_WIN
    # pendiente por el origen de la caída contra la exposición (media por sujeto, condiciones K>0)
    D = C[C.K > 0]
    for dec in ("csp", "net"):
        x, y = D.p_touched_med_subj.to_numpy(float), D[f"dBA_mean_{dec}"].to_numpy(float)
        C.loc[C.K > 0, f"pendiente_origen_{dec}"] = float((x * y).sum() / (x * x).sum())
        rho = sps.spearmanr(x, y)
        C.loc[C.K > 0, f"spearman_rho_{dec}"] = rho.statistic
    # unir pruebas
    for dec, name in (("csp", "CSP+LDA"), ("net", "EEGNet")):
        t = T[T.decoder == name].set_index("K")
        for i, lab in enumerate(SWEEP):
            if lab == "ref":
                continue
            C.loc[i, f"wilcoxon_p_{dec}"] = t.loc[SWEEP_K[lab], "wilcoxon_p"]
            C.loc[i, f"p_holm_{dec}"] = t.loc[SWEEP_K[lab], "p_holm"]
            C.loc[i, f"r_{dec}"] = t.loc[SWEEP_K[lab], "r"]
            C.loc[i, f"friedman_p_{dec}"] = t.loc[SWEEP_K[lab], "friedman_p"]
    C.to_csv(os.path.join(OUT, "m3c_exp_cond.csv"), index=False)

    # ------------- emparejamiento por ensayo con la referencia
    Dp = paired_trials(ex, NEW, NET_NEW, NEW, NET_NEW)
    Dp = Dp[Dp.label_cond != "ref"].copy()
    Dp.to_csv(os.path.join(OUT, "m3c_exp_trials.csv"), index=False)
    crow = []
    for lab in SWEEP[1:] + ["TODAS"]:
        g = Dp if lab == "TODAS" else Dp[Dp.label_cond == lab]
        for tag, name in (("csp", "CSP+LDA"), ("net", "EEGNet")):
            for tc, tname in ((True, "tocado"), (False, "no tocado")):
                h = g[g.touched == tc]
                k, n = int(h[f"chg_{tag}"].sum()), len(h)
                lo, hi = wilson(k, n)
                kv = int(((h[f"valid_{tag}"] == 1) & (h[f"ref_valid_{tag}"] == 1) & (h[f"pred_{tag}"] != h[f"ref_pred_{tag}"])).sum())
                crow.append(dict(label=lab, decoder=name, grupo=tname, n_ensayos=n, n_cambios=k, tasa_cambio=k / n if n else np.nan,
                                 ic_lo=lo, ic_hi=hi, n_perdidas=int(((h[f"ref_ok_{tag}"] == 1) & (h[f"ok_{tag}"] == 0)).sum()),
                                 n_ganadas=int(((h[f"ref_ok_{tag}"] == 0) & (h[f"ok_{tag}"] == 1)).sum()),
                                 acc_cond=h[f"ok_{tag}"].mean() if n else np.nan, acc_ref=h[f"ref_ok_{tag}"].mean() if n else np.nan))
    X = pd.DataFrame(crow)
    X.to_csv(os.path.join(OUT, "m3c_exp_cambio.csv"), index=False)

    # cambio de decisión según muestras recibidas en la ventana (dosis); 'no tocado' = 1000 muestras en la ventana
    bins = [-1, 124, 499, 749, 899, 999, 10**6]
    labs = ["inválido (<125)", "125-499", "500-749", "750-899", "900-999", ">=1000"]
    Dp["bin"] = pd.cut(Dp.n_samples, bins=bins, labels=labs)
    dose = []
    for b, g in Dp.groupby("bin", observed=True):
        for tag, name in (("csp", "CSP+LDA"), ("net", "EEGNet")):
            k, n = int(g[f"chg_{tag}"].sum()), len(g)
            lo, hi = wilson(k, n)
            dose.append(dict(bin=b, decoder=name, n_ensayos=n, n_cambios=k, tasa_cambio=k / n, ic_lo=lo, ic_hi=hi,
                             n_perdidas=int(((g[f"ref_ok_{tag}"] == 1) & (g[f"ok_{tag}"] == 0)).sum()),
                             n_ganadas=int(((g[f"ref_ok_{tag}"] == 0) & (g[f"ok_{tag}"] == 1)).sum()),
                             acc_cond=g[f"ok_{tag}"].mean(), acc_ref=g[f"ref_ok_{tag}"].mean()))
    pd.DataFrame(dose).to_csv(os.path.join(OUT, "m3c_exp_dosis.csv"), index=False)
    print(pd.DataFrame(dose).round(4).to_string(index=False))
    # ensayos no tocados que cambiaron: distancia (s) del hueco más cercano a la ventana de decodificación
    # (el consumidor filtra con 1 s de relleno a cada lado de la ventana: consumer.py, PAD_S)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    print(C[["label", "K", "n_exec", "p_touched_med_subj", "p_touched_q1", "p_touched_q3", "p_touched_glob", "E_modelo",
             "k_touched_total", "n_trials_total", "k_invalid_total", "lost_frac_glob"]].round(4).to_string(index=False))
    print(C[["label", "bacc_csp_med", "dBA_med_csp", "dBA_mean_csp", "n_subj_peor_csp", "n_subj_mejor_csp", "wilcoxon_p_csp", "p_holm_csp", "r_csp",
             "friedman_p_csp", "pendiente_origen_csp", "spearman_rho_csp"]].round(4).to_string(index=False))
    print(C[["label", "bacc_net_med", "dBA_med_net", "dBA_mean_net", "n_subj_peor_net", "n_subj_mejor_net", "wilcoxon_p_net", "p_holm_net", "r_net",
             "friedman_p_net", "pendiente_origen_net", "spearman_rho_net"]].round(4).to_string(index=False))
    print(X.round(4).to_string(index=False))

    # ------------- figura
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for dec, mk, ls in (("csp", "o", "-"), ("net", "s", "--")):
        x = C.p_touched_med_subj.to_numpy(float) * 100
        y = C[f"dBA_med_{dec}"].fillna(0).to_numpy(float) * 100
        ym = C[f"dBA_mean_{dec}"].fillna(0).to_numpy(float) * 100
        ax[0].plot(x, ym, ls, marker=mk, color="k", mfc="w" if dec == "net" else "k", label="CSP+LDA" if dec == "csp" else "EEGNet")
    ax[0].axhline(0, color="gray", lw=.6)
    for _, r in C.iterrows():
        ax[0].annotate(f"K={int(r.K)}", (r.p_touched_med_subj * 100, r.dBA_mean_csp * 100 if r.K else 0), textcoords="offset points",
                       xytext=(4, 5), fontsize=8)
    ax[0].set_xlabel("Ensayos tocados (%; mediana entre sujetos)"); ax[0].set_ylabel("Δ balanced accuracy vs. ref. (pp, media)")
    ax[0].legend(frameon=False)
    for tag, name, mk in (("csp", "CSP+LDA", "o"), ("net", "EEGNet", "s")):
        for lab, off in zip(SWEEP[1:], (-0.12, -0.04, .04, .12)):
            h = X[(X.label == lab) & (X.decoder == name) & (X.grupo == "tocado")]
        pts = X[(X.label.isin(SWEEP[1:])) & (X.decoder == name) & (X.grupo == "tocado")]
        ax[1].plot([SWEEP_K[l] for l in pts.label], pts.tasa_cambio * 100, "-" if tag == "csp" else "--", marker=mk, color="k",
                   mfc="k" if tag == "csp" else "w", label=name)
    ax[1].set_xticks([2, 5, 10, 20]); ax[1].set_xlabel("Cortes de 1 s por corrida (K)"); ax[1].set_ylabel("Cambio de decisión en tocados (%)")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "m3c_exp_curva.png"), dpi=200)


if __name__ == "__main__":
    main()
