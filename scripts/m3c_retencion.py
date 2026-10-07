"""M3c · Tarea 3: RETENCIÓN (hold_trial, campaña nueva) vs BORRADO (burst_trial, campaña anterior), mismas severidades.

Mismo patrón de tramo contiguo dentro de la ventana 2-6 s de cada ensayo (10/25/40 % de la ventana):
  burst_trial : las muestras del tramo se borran (el receptor no las recibe)
  hold_trial  : las muestras del tramo se reemplazan por la última muestra válida (todas llegan; n_samples = 1000)
Cada condición se compara contra SU referencia (hold vs ref de la campaña nueva; burst vs ref de la campaña anterior):
unidad = sujeto (mediana de 5 corridas), Friedman + Wilcoxon apareado + Holm dentro del tipo + r (bcibench.stats).
Comparación directa hold vs burst: Wilcoxon apareado por sujeto sobre la diferencia contra la referencia
(Δhold - Δburst), Holm sobre las 3 severidades.
"Tocado" en hold = el interval_planned de fault_log.jsonl cae (se solapa) en la ventana [onset+2, onset+6] s.
En burst no hay interval_planned en fault_log (queda vacío): tocado = n_samples < 1000 o inválido.

Salidas (campaign-m3/analysis/): m3c_ret_exec.csv, m3c_ret_tests.csv, m3c_ret_cond.csv, m3c_ret_directo.csv,
m3c_ret_cambio.csv, m3c_ret_cambio_sujeto.csv, m3c_ret_trials.csv, m3c_ret_fig.png
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from m3c_common import (NEW, NET_NEW, NET_OLD, OLD, OLD_ROOT, OUT, BURST, HOLD, lab_filter, load, paired_trials,
                        subject_med, tests_group, wilcoxon_diff, wilson)
from bcibench.stats import holm

SEV = {"hold_trial-0.1": 0.1, "hold_trial-0.25": 0.25, "hold_trial-0.4": 0.4,
       "burst_trial-0.1": 0.1, "burst_trial-0.25": 0.25, "burst_trial-0.4": 0.4}


def exec_table(raw, labels, net_file):
    keep = lab_filter(labels)
    a, tr, _ = load(raw, "trials.csv", keep)
    b, _, _ = load(raw, net_file, keep)
    return a.merge(b[["exec_id", "bacc"]].rename(columns={"bacc": "bacc_net"}), on="exec_id"), tr


def main():
    ex_h, tr_h = exec_table(NEW, HOLD, NET_NEW)
    ex_b, tr_b = exec_table(OLD, ["ref"] + BURST, NET_OLD)
    ex_h["campania"], ex_b["campania"] = "nueva", "anterior"
    assert len(ex_h) == 180 and len(ex_b) == 180, (len(ex_h), len(ex_b))
    EX = pd.concat([ex_h, ex_b], ignore_index=True)
    EX.to_csv(os.path.join(OUT, "m3c_ret_exec.csv"), index=False)

    cols = ["bacc", "bacc_net"]
    med_h, med_b = subject_med(ex_h, cols), subject_med(ex_b, cols)
    tests = []
    for med, kind in ((med_h, "hold_trial"), (med_b, "burst_trial")):
        for m, name in (("bacc", "CSP+LDA"), ("bacc_net", "EEGNet")):
            t = tests_group(med, m, kind)
            t.insert(0, "decoder", name)
            tests.append(t)
    T = pd.concat(tests, ignore_index=True)
    T.to_csv(os.path.join(OUT, "m3c_ret_tests.csv"), index=False)
    # control de consistencia con el estudio: p de burst_trial CSP+LDA y EEGNet vs analysis/t_desempeno.csv y analysis-eegnet
    for folder, name in (("analysis", "CSP+LDA"), ("analysis-eegnet", "EEGNet")):
        old = pd.read_csv(os.path.join(OLD_ROOT, folder, "t_desempeno.csv"))
        old = old[old.kind == "burst_trial"].set_index("severity")
        mine = T[(T.kind == "burst_trial") & (T.decoder == name)].set_index("severity")
        print(f"[control] burst_trial {name}: p_holm estudio vs recalculado:",
              [(s, round(old.loc[s, "p_holm"], 4), round(mine.loc[s, "p_holm"], 4)) for s in mine.index],
              "| diff estudio vs recalculado:", [(s, round(old.loc[s, "diff"], 4), round(mine.loc[s, "diff"], 4)) for s in mine.index])

    # ------------- Δ por sujeto, tabla por condición
    def delta(med, kind, col):
        ref = med[med.kind == "none"].set_index("subject")[col]
        out = {}
        for sev in sorted(med[med.kind == kind].severity.unique()):
            s = med[(med.kind == kind) & (med.severity == sev)].set_index("subject")[col]
            out[sev] = (s - ref).dropna()
        return out

    rows = []
    for fam, med, kind in (("retención (hold)", med_h, "hold_trial"), ("borrado (burst)", med_b, "burst_trial")):
        for col, name in (("bacc", "CSP+LDA"), ("bacc_net", "EEGNet")):
            ref_med = med[med.kind == "none"][col].median()
            dd = delta(med, kind, col)
            for sev, d in dd.items():
                s = med[(med.kind == kind) & (med.severity == sev)][col]
                t = T[(T.kind == kind) & (T.decoder == name) & np.isclose(T.severity, sev)].iloc[0]
                rows.append(dict(familia=fam, decoder=name, severidad=sev, n_subj=len(d), bacc_ref_med=ref_med,
                                 bacc_cond_med=s.median(), dBA_med=d.median(), dBA_mean=d.mean(), dBA_min=d.min(), dBA_max=d.max(),
                                 n_peor=int((d < 0).sum()), n_igual=int((d == 0).sum()), n_mejor=int((d > 0).sum()),
                                 wilcoxon_p=t.wilcoxon_p, p_holm=t.p_holm, r=t.r, friedman_p=t.friedman_p))
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(OUT, "m3c_ret_cond.csv"), index=False)

    # ------------- hold vs burst directo (Δ contra su referencia), por sujeto
    dr = []
    for col, name in (("bacc", "CSP+LDA"), ("bacc_net", "EEGNet")):
        dh, db = delta(med_h, "hold_trial", col), delta(med_b, "burst_trial", col)
        tt = [wilcoxon_diff(dh[s], db[s]) for s in (0.1, 0.25, 0.4)]
        ph = holm([t["p"] if not np.isnan(t["p"]) else 1.0 for t in tt])
        for s, t, p in zip((0.1, 0.25, 0.4), tt, ph):
            dr.append(dict(decoder=name, severidad=s, n=t["n"], dBA_hold_med=dh[s].median(), dBA_burst_med=db[s].median(),
                           dif_hold_menos_burst_med=(dh[s] - db[s]).median(), wilcoxon_p=t["p"], p_holm=p, r=t["r"],
                           n_hold_menos_caida=int((dh[s] > db[s]).sum()), n_hold_mas_caida=int((dh[s] < db[s]).sum())))
    DR = pd.DataFrame(dr)
    DR.to_csv(os.path.join(OUT, "m3c_ret_directo.csv"), index=False)

    # ------------- ensayos emparejados con su referencia
    Dh = paired_trials(ex_h[ex_h.label != "ref"], NEW, NET_NEW, NEW, NET_NEW, with_intervals=True)
    Db = paired_trials(ex_b[ex_b.label != "ref"], OLD, NET_OLD, OLD, NET_OLD, with_intervals=False)
    Dh["familia"], Db["familia"] = "hold", "burst"
    # tocado en hold = solape con interval_planned; en burst = n_samples < 1000 o inválido
    Dh["tocado"] = Dh.iv_overlap > 0
    Db["tocado"] = Db.touched
    print("\nhold: ensayos con >=1 interval_planned en la ventana:", int((Dh.iv_overlap > 0).sum()), "de", len(Dh),
          "| intervalo totalmente dentro de la ventana:", int((Dh.iv_inside == 1).sum()),
          "| solape medio (muestras) por severidad:", Dh.groupby("severity").iv_overlap.mean().round(1).to_dict(),
          "| n_samples<1000:", int((Dh.n_samples < 1000).sum()), "| n_intervals por ejecución (único):", sorted(Dh.n_intervals.unique()))
    print("burst: ensayos con n_samples<1000:", int((Db.n_samples < 1000).sum()), "de", len(Db),
          "| inválidos:", int((Db.valid_online == 0).sum()), "| n_samples medio por severidad:",
          Db.groupby("severity").n_samples.mean().round(1).to_dict())
    A = pd.concat([Dh, Db], ignore_index=True)
    A.to_csv(os.path.join(OUT, "m3c_ret_trials.csv"), index=False)

    crow = []
    for fam in ("hold", "burst"):
        for sev in (0.1, 0.25, 0.4, "todas"):
            g = A[(A.familia == fam)] if sev == "todas" else A[(A.familia == fam) & np.isclose(A.severity, sev)]
            for tag, name in (("csp", "CSP+LDA"), ("net", "EEGNet")):
                for tc, tn in ((True, "tocado"), (False, "no tocado")):
                    h = g[g.tocado == tc]
                    n = len(h)
                    if n == 0:
                        continue
                    k = int(h[f"chg_{tag}"].sum())
                    lo, hi = wilson(k, n)
                    crow.append(dict(familia=fam, severidad=sev, decoder=name, grupo=tn, n_ensayos=n, n_cambios=k,
                                     tasa_cambio=k / n, ic_lo=lo, ic_hi=hi,
                                     n_perdidas=int(((h[f"ref_ok_{tag}"] == 1) & (h[f"ok_{tag}"] == 0)).sum()),
                                     n_ganadas=int(((h[f"ref_ok_{tag}"] == 0) & (h[f"ok_{tag}"] == 1)).sum()),
                                     acc_cond=h[f"ok_{tag}"].mean(), acc_ref=h[f"ref_ok_{tag}"].mean()))
    X = pd.DataFrame(crow)
    X.to_csv(os.path.join(OUT, "m3c_ret_cambio.csv"), index=False)

    # tasa de cambio por sujeto (media sobre sus ensayos) -> Wilcoxon hold vs burst por severidad (unidad = sujeto)
    S_ = (A.groupby(["familia", "severity", "subject"])[["chg_csp", "chg_net"]].mean().reset_index())
    S_.to_csv(os.path.join(OUT, "m3c_ret_cambio_sujeto.csv"), index=False)
    srows = []
    for tag, name in (("chg_csp", "CSP+LDA"), ("chg_net", "EEGNet")):
        tt, sv = [], (0.1, 0.25, 0.4)
        for s in sv:
            a = S_[(S_.familia == "hold") & np.isclose(S_.severity, s)].set_index("subject")[tag]
            b = S_[(S_.familia == "burst") & np.isclose(S_.severity, s)].set_index("subject")[tag]
            tt.append((a, b, wilcoxon_diff(a, b)))
        ph = holm([t[2]["p"] if not np.isnan(t[2]["p"]) else 1.0 for t in tt])
        for s, (a, b, t), p in zip(sv, tt, ph):
            srows.append(dict(decoder=name, severidad=s, n=t["n"], cambio_hold_med_subj=a.median(), cambio_burst_med_subj=b.median(),
                              dif_med=(a - b).median(), wilcoxon_p=t["p"], p_holm=p, r=t["r"]))
    SR = pd.DataFrame(srows)
    SR.to_csv(os.path.join(OUT, "m3c_ret_cambio_directo.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    print("\n== tests vs referencia propia ==")
    print(T[["decoder", "kind", "severity", "n", "median_ref", "median_cond", "diff", "friedman_p", "wilcoxon_p", "p_holm", "r"]].round(4).to_string(index=False))
    print("\n== por condición ==")
    print(C.round(4).to_string(index=False))
    print("\n== hold vs burst directo (Δ contra su referencia) ==")
    print(DR.round(4).to_string(index=False))
    print("\n== cambio de decisión por ensayo ==")
    print(X.round(4).to_string(index=False))
    print("\n== cambio por sujeto, hold vs burst ==")
    print(SR.round(4).to_string(index=False))

    # ------------- figura
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for a_, name in zip(ax, ("CSP+LDA", "EEGNet")):
        for fam, mk, ls, mfc in (("retención (hold)", "o", "-", "k"), ("borrado (burst)", "s", "--", "w")):
            g = C[(C.familia == fam) & (C.decoder == name)]
            a_.plot(g.severidad * 100, g.dBA_mean * 100, ls, marker=mk, color="k", mfc=mfc, label=fam)
        a_.axhline(0, color="gray", lw=.6)
        a_.set_title(name); a_.set_xlabel("Tramo afectado de la ventana (%)"); a_.set_xticks([10, 25, 40])
    ax[0].set_ylabel("Δ balanced accuracy vs. ref. (pp, media)")
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "m3c_ret_fig.png"), dpi=200)


if __name__ == "__main__":
    main()
