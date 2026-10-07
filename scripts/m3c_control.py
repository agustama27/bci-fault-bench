"""M3c · Tarea 5: control de máquina. Referencia de la campaña nueva vs referencia de la campaña anterior (mismas 45 corridas:
9 sujetos x 5 corridas, misma señal, mismo decodificador congelado).

Por ensayo (24 por corrida): decisión idéntica = ambos válidos y misma predicción, o ambos inválidos. EEGNet: 'ceros' de la
campaña nueva (m3_eegnet_relleno.py) vs trials_eegnet.csv del Entregable 2 (el modo 'ceros' reproduce ese archivo en el 99,96 %
de las decisiones de las 855 ejecuciones, results/vm/analysis-m3-eegnet/por_ejecucion.csv, columna reproduce_e2).
Por ejecución: latencia media y demás variables de telemetría; Wilcoxon apareado sobre la mediana por sujeto (unidad = sujeto).

Salidas (campaign-m3/analysis/): m3c_ctl_ensayos.csv, m3c_ctl_resumen.csv, m3c_ctl_exec.csv, m3c_ctl_tests.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from m3c_common import NEW, NET_NEW, NET_OLD, OLD, OUT, lab_filter, load, wilcoxon_diff


def main():
    keep = lab_filter(["ref"])
    en, tn, _ = load(NEW, "trials.csv", keep)
    eo, to, _ = load(OLD, "trials.csv", keep)
    en2, _, _ = load(NEW, NET_NEW, keep)
    eo2, _, _ = load(OLD, NET_OLD, keep)
    assert len(en) == len(eo) == 45
    key = ["subject", "run"]
    E = en.merge(eo, on=key, suffixes=("_new", "_old"))
    E2 = en2[key + ["bacc"]].merge(eo2[key + ["bacc"]], on=key, suffixes=("_new", "_old"))
    assert len(E) == 45

    # ---------------- decisiones por ensayo
    rows = []
    for _, e in E.iterrows():
        d_n = os.path.join(NEW, f"s{int(e.subject):02d}-r{e.run}-ref")
        d_o = os.path.join(OLD, f"s{int(e.subject):02d}-r{e.run}-ref")
        for dec, fn, fo in (("CSP+LDA", "trials.csv", "trials.csv"), ("EEGNet", NET_NEW, NET_OLD)):
            a, b = pd.read_csv(os.path.join(d_n, fn)), pd.read_csv(os.path.join(d_o, fo))
            assert len(a) == len(b) == 24 and np.abs(a.onset - b.onset).max() < 1e-6
            va, vb = a.valid == 1, b.valid == 1
            same = ((va & vb & (pd.to_numeric(a.pred, errors="coerce") == pd.to_numeric(b.pred, errors="coerce"))) | (~va & ~vb))
            rows.append(dict(subject=int(e.subject), run=e.run, decoder=dec, n=24, n_identicas=int(same.sum()),
                             n_valid_new=int(va.sum()), n_valid_old=int(vb.sum()),
                             n_samples_iguales=int((a.n_samples == b.n_samples).sum()),
                             dif_proba_media=float((pd.to_numeric(a.proba, errors="coerce") - pd.to_numeric(b.proba, errors="coerce")).abs().mean())))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "m3c_ctl_ensayos.csv"), index=False)
    summ = []
    for dec, g in R.groupby("decoder"):
        per_subj = g.groupby("subject")[["n_identicas", "n"]].sum()
        summ.append(dict(decoder=dec, ejecuciones=len(g), ensayos=int(g.n.sum()), identicas=int(g.n_identicas.sum()),
                         pct_identicas=100 * g.n_identicas.sum() / g.n.sum(), ejec_con_alguna_diferencia=int((g.n_identicas < g.n).sum()),
                         peor_ejecucion_pct=100 * (g.n_identicas / g.n).min(), n_samples_iguales_pct=100 * g.n_samples_iguales.sum() / g.n.sum(),
                         dif_proba_abs_media=g.dif_proba_media.mean(), dif_proba_abs_max_ejec=g.dif_proba_media.max(),
                         pct_identicas_por_sujeto=str({int(s): round(float(100 * r.n_identicas / r.n), 1) for s, r in per_subj.iterrows()})))
    SM = pd.DataFrame(summ)
    SM.to_csv(os.path.join(OUT, "m3c_ctl_resumen.csv"), index=False)

    # ---------------- variables por ejecución: nueva vs anterior
    cols = ["lat_mean_ms", "lat_max_ms", "ia_std_ms", "ia_max_ms", "recv_ratio", "decision_delay_ms", "bacc", "valid_rate", "timing_p99_ms"]
    rows, tests = [], []
    for c in cols:
        a = E.set_index(["subject", "run"])[f"{c}_new"]
        b = E.set_index(["subject", "run"])[f"{c}_old"]
        sa = a.groupby("subject").median()
        sb = b.groupby("subject").median()
        t = wilcoxon_diff(sa, sb)
        rows.append(dict(variable=c, nueva_med_subj=sa.median(), anterior_med_subj=sb.median(), dif_med_subj=(sa - sb).median(),
                         dif_media_ejec=float((a - b).mean()), dif_ejec_min=float((a - b).min()), dif_ejec_max=float((a - b).max()),
                         n_subj=t["n"], wilcoxon_p=t["p"], r=t["r"], n_subj_mayor_nueva=int((sa > sb).sum()), n_subj_menor_nueva=int((sa < sb).sum())))
    x = E2.set_index(["subject", "run"])
    sa, sb = x.bacc_new.groupby("subject").median(), x.bacc_old.groupby("subject").median()
    t = wilcoxon_diff(sa, sb)
    rows.append(dict(variable="bacc EEGNet (ceros vs E2)", nueva_med_subj=sa.median(), anterior_med_subj=sb.median(), dif_med_subj=(sa - sb).median(),
                     dif_media_ejec=float((x.bacc_new - x.bacc_old).mean()), dif_ejec_min=float((x.bacc_new - x.bacc_old).min()),
                     dif_ejec_max=float((x.bacc_new - x.bacc_old).max()), n_subj=t["n"], wilcoxon_p=t["p"], r=t["r"],
                     n_subj_mayor_nueva=int((sa > sb).sum()), n_subj_menor_nueva=int((sa < sb).sum())))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "m3c_ctl_tests.csv"), index=False)
    E.to_csv(os.path.join(OUT, "m3c_ctl_exec.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(SM.round(4).to_string(index=False))
    print(T.round(5).to_string(index=False))
    print(R[R.n_identicas < R.n].to_string(index=False))


if __name__ == "__main__":
    main()
