"""M3c · Tarea 4: fallas silenciosas con hold_trial (campaña exploratoria post hoc) vs burst_trial (campaña anterior).

MISMA definición que el estudio (bcibench.metrics):
  operativo (por segundo) = llegaron muestras (n_samples > 0) y exceptions == 0
  degradado (por segundo) = balanced accuracy móvil W = 8 ensayos (rolling_bacc; ensayo inválido = error) de CSP+LDA en línea
                            < umbral del sujeto (thresholds.json de la CAMPAÑA ANTERIOR: mínimo de la móvil en sus 5 referencias)
  silenciosa = operativo y degradado; proporción sobre los segundos evaluables (con móvil disponible), como divergence().
Se informa la mediana entre sujetos de la proporción por ejecución (mediana de sus 5 corridas) y el conteo total.

Controles: (a) la referencia nueva no debe cruzar el umbral (se informa cuántos segundos lo hacen); (b) con las funciones
recalculadas sobre la campaña anterior se reproduce analysis/windows.csv (degraded) y t_divergencia.csv.
Telemetría: pruebas ref vs hold sobre las variables de infraestructura (sujeto = mediana de corridas) y alarma del detector por
umbrales del estudio (límites de la referencia de los OTROS sujetos, LOSO) sobre segundos de hold, de ref y de burst.

Salidas (campaign-m3/analysis/): m3c_sil_exec.csv, m3c_sil_cond.csv, m3c_sil_tests.csv, m3c_sil_directo.csv,
m3c_sil_telemetria.csv, m3c_sil_detector.csv, m3c_sil_auc.csv, m3c_sil_latencia_condiciones.csv, m3c_sil_control.csv
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from m3c_common import (BURST, HOLD, NEW, OLD, OLD_ROOT, OUT, SFREQ, S, M, lab_filter, load, load_events, subject_med,
                        tests_group, wilcoxon_diff)
from bcibench.stats import holm

W = 8
THR = {int(k): v for k, v in json.load(open(os.path.join(OLD_ROOT, "analysis", "thresholds.json")))["thresholds"].items()}
INFRA = ["recv_ratio", "n_gaps", "gap_s", "lat_mean_ms", "lat_max_ms", "ia_std_ms", "ia_max_ms", "secs_no_data"]


def windows_for(ex, trials, tele):
    ws = []
    for _, e in ex.iterrows():
        roll = M.rolling_bacc(trials[e.exec_id], W)
        w = M.window_table(tele[e.exec_id], roll, THR[int(e.subject)])
        w["exec_id"], w["subject"], w["run"], w["kind"], w["severity"], w["label"] = e.exec_id, e.subject, e.run, e.kind, e.severity, e.label
        ws.append(w)
    return pd.concat(ws, ignore_index=True)


def exec_summary(Wt):
    rows = []
    for (eid, s, run, kind, sev, lab), g in Wt.groupby(["exec_id", "subject", "run", "kind", "severity", "label"]):
        v = g.dropna(subset=["degraded"])
        n = len(v)
        silent = int(((v.operational == 1) & (v.degraded == 1)).sum())
        loud = int(((v.operational == 0) & (v.degraded == 0)).sum())
        deg = int((v.degraded == 1).sum())
        rows.append(dict(exec_id=eid, subject=int(s), run=run, kind=kind, severity=sev, label=lab, n_secs=len(g), n_eval=n,
                         n_operativo=int(v.operational.sum()), n_degradado=deg, n_silenciosa=silent, n_ruidosa=loud,
                         p_silenciosa=silent / n if n else np.nan, p_degradado=deg / n if n else np.nan,
                         p_operativo=float(v.operational.mean()) if n else np.nan,
                         secs_sin_muestras=int((g.n_samples == 0).sum()), secs_con_excepcion=int((g.exceptions > 0).sum())))
    return pd.DataFrame(rows)


def main():
    # ---------------- campaña nueva (todas las condiciones) y anterior (ref + burst)
    ex_n, tr_n, te_n = load(NEW, "trials.csv")
    ex_o, tr_o, te_o = load(OLD, "trials.csv", lab_filter(["ref"] + BURST))
    W_n, W_o = windows_for(ex_n, tr_n, te_n), windows_for(ex_o, tr_o, te_o)
    E_n, E_o = exec_summary(W_n), exec_summary(W_o)
    E_n["campania"], E_o["campania"] = "nueva", "anterior"
    E = pd.concat([E_n, E_o], ignore_index=True)
    E.to_csv(os.path.join(OUT, "m3c_sil_exec.csv"), index=False)

    # ---------------- controles
    ctl = []
    # (a) referencias: segundos degradados
    for nm, Wt in (("ref nueva", W_n[W_n.kind == "none"]), ("ref anterior", W_o[W_o.kind == "none"])):
        v = Wt.dropna(subset=["degraded"])
        ctl.append(dict(control=f"{nm}: segundos evaluables / degradados / min(rbacc) - umbral", valor=f"{len(v)} / {int((v.degraded == 1).sum())}"))
        mn = v.groupby("subject").rbacc.min()
        ctl.append(dict(control=f"{nm}: min de la móvil por sujeto (>= umbral)",
                        valor=str({int(s): (round(float(mn[s]), 3), round(THR[int(s)], 3)) for s in mn.index})))
    # (b) reproducción de windows.csv de la campaña anterior
    old_w = pd.read_csv(os.path.join(OLD_ROOT, "analysis", "windows.csv"))
    old_w = old_w[old_w.label.isin(["ref"] + BURST)]
    mrg = W_o.merge(old_w[["exec_id", "sec", "degraded", "operational"]], on=["exec_id", "sec"], suffixes=("", "_est"))
    same_deg = ((mrg.degraded == mrg.degraded_est) | (mrg.degraded.isna() & mrg.degraded_est.isna())).mean()
    ctl.append(dict(control="reproduce windows.csv (anterior, ref+burst): segundos comparados / % degraded idéntico / % operational idéntico",
                    valor=f"{len(mrg)} / {100 * same_deg:.3f} / {100 * (mrg.operational == mrg.operational_est).mean():.3f}"))
    pd.DataFrame(ctl).to_csv(os.path.join(OUT, "m3c_sil_control.csv"), index=False)
    for c in ctl:
        print("[control]", c["control"], "->", c["valor"])

    # ---------------- por condición
    def cond_table(Eg, labels, ref_label="ref"):
        rows = []
        for lab in labels:
            g = Eg[Eg.label == lab]
            sm = g.groupby("subject")[["p_silenciosa", "p_degradado", "p_operativo"]].median()
            rows.append(dict(label=lab, n_exec=len(g), n_subj=len(sm),
                             p_silenciosa_med_subj=sm.p_silenciosa.median(), p_silenciosa_q1=sm.p_silenciosa.quantile(.25),
                             p_silenciosa_q3=sm.p_silenciosa.quantile(.75), p_silenciosa_max_subj=sm.p_silenciosa.max(),
                             p_degradado_med_subj=sm.p_degradado.median(), p_operativo_med_subj=sm.p_operativo.median(),
                             secs_eval_total=int(g.n_eval.sum()), secs_degradados_total=int(g.n_degradado.sum()),
                             secs_silenciosos_total=int(g.n_silenciosa.sum()),
                             frac_degradado_que_es_silencioso=(g.n_silenciosa.sum() / g.n_degradado.sum()) if g.n_degradado.sum() else np.nan,
                             p_silenciosa_glob=g.n_silenciosa.sum() / g.n_eval.sum(),
                             ejec_con_silenciosa=int((g.n_silenciosa > 0).sum()),
                             subj_con_silenciosa=int((g.groupby("subject").n_silenciosa.sum() > 0).sum()),
                             subj_con_silenciosa_lista=str(sorted(int(x) for x in g.groupby("subject").n_silenciosa.sum().loc[lambda z: z > 0].index)),
                             secs_sin_muestras_total=int(g.secs_sin_muestras.sum()), secs_con_excepcion_total=int(g.secs_con_excepcion.sum())))
        return pd.DataFrame(rows)

    lab_new = sorted(E_n.label.unique(), key=lambda s: (s != "ref", s))
    CN, CO = cond_table(E_n, lab_new), cond_table(E_o, ["ref"] + BURST)
    CN["campania"], CO["campania"] = "nueva", "anterior"
    C = pd.concat([CN, CO], ignore_index=True)
    C.to_csv(os.path.join(OUT, "m3c_sil_cond.csv"), index=False)

    # ---------------- pruebas: p_silenciosa y p_degradado, ref vs hold; ref vs burst (misma lógica por tipo)
    tests = []
    for nm, Eg, kind in (("hold_trial", E_n, "hold_trial"), ("burst_trial", E_o, "burst_trial")):
        med = subject_med(Eg, ["p_silenciosa", "p_degradado"])
        for m in ("p_silenciosa", "p_degradado"):
            t = tests_group(med, m, kind)
            t.insert(0, "metric", m)
            tests.append(t)
    TT = pd.concat(tests, ignore_index=True)
    TT.to_csv(os.path.join(OUT, "m3c_sil_tests.csv"), index=False)
    # hold vs burst directo (proporciones por sujeto; las referencias no divergen -> comparación de proporciones absolutas)
    med_h, med_b = subject_med(E_n, ["p_silenciosa", "p_degradado"]), subject_med(E_o, ["p_silenciosa", "p_degradado"])
    dr = []
    for m in ("p_silenciosa", "p_degradado"):
        tt = []
        for s in (0.1, 0.25, 0.4):
            a = med_h[(med_h.kind == "hold_trial") & np.isclose(med_h.severity, s)].set_index("subject")[m]
            b = med_b[(med_b.kind == "burst_trial") & np.isclose(med_b.severity, s)].set_index("subject")[m]
            tt.append((a, b, wilcoxon_diff(a, b)))
        ph = holm([t[2]["p"] if not np.isnan(t[2]["p"]) else 1.0 for t in tt])
        for s, (a, b, t), p in zip((0.1, 0.25, 0.4), tt, ph):
            dr.append(dict(metric=m, severidad=s, n=t["n"], hold_med_subj=a.median(), burst_med_subj=b.median(),
                           dif_med=(a - b).median(), wilcoxon_p=t["p"], p_holm=p, r=t["r"]))
    DR = pd.DataFrame(dr)
    DR.to_csv(os.path.join(OUT, "m3c_sil_directo.csv"), index=False)

    # ---------------- telemetría: ref vs hold (variables de infraestructura por ejecución)
    med_i = subject_med(ex_n[ex_n.label.isin(HOLD)], INFRA)
    rows = []
    for m in INFRA:
        t = tests_group(med_i, m, "hold_trial")
        t.insert(0, "metric", m)
        rows.append(t)
    TI = pd.concat(rows, ignore_index=True)
    TI.to_csv(os.path.join(OUT, "m3c_sil_telemetria.csv"), index=False)

    # ---------------- detector por umbrales del estudio (LOSO) sobre segundos
    FE = S.FEATURES

    def alarm_rates(Wt, nm):
        V = Wt.dropna(subset=["degraded"]).copy()
        V["alarm"] = False
        for s in V.subject.unique():
            lim = S.reference_limits(V[(V.subject != s) & (V.kind == "none")])
            m = V.subject == s
            V.loc[m, "alarm"] = S.threshold_detector(V[m], lim)
        V["campania"] = nm
        return V

    Vn, Vo = alarm_rates(W_n, "nueva"), alarm_rates(W_o, "anterior")
    # segundos con el tramo retenido dentro del segundo (interval_planned en muestras -> segundos)
    ivsec = {}
    for eid in Vn[Vn.kind == "hold_trial"].exec_id.unique():
        ev = load_events(os.path.join(NEW, eid, "fault_log.jsonl"))
        secs = set()
        for e in ev:
            if e.get("event") == "interval_planned":
                secs.update(range(int(e["start_sample"] // SFREQ), int((e["end_sample"] - 1) // SFREQ) + 1))
        ivsec[eid] = secs
    Vn["en_tramo"] = [(s in ivsec.get(e, ())) for e, s in zip(Vn.exec_id, Vn.sec)]
    det = []

    def row(nm, V, extra=""):
        n = len(V)
        d = V[V.degraded == 1]
        sil = V[(V.operational == 1) & (V.degraded == 1)]
        det.append(dict(grupo=nm, n_seg=n, tasa_alarma=float(V.alarm.mean()) if n else np.nan, n_alarmas=int(V.alarm.sum()),
                        n_degradados=len(d), alarmas_en_degradados=int(d.alarm.sum()),
                        recall_degradados=float(d.alarm.mean()) if len(d) else np.nan,
                        n_silenciosos=len(sil), alarmas_en_silenciosos=int(sil.alarm.sum())))
    row("ref nueva (LOSO)", Vn[Vn.kind == "none"])
    for lab in HOLD[1:]:
        g = Vn[Vn.label == lab]
        row(f"{lab} (todos los segundos)", g)
        row(f"{lab} (segundos que contienen el tramo retenido)", g[g.en_tramo])
        row(f"{lab} (segundos sin tramo)", g[~g.en_tramo])
    row("hold (3 severidades, todos los segundos)", Vn[Vn.kind == "hold_trial"])
    row("hold (segundos con tramo)", Vn[(Vn.kind == "hold_trial") & Vn.en_tramo])
    row("ref anterior (LOSO)", Vo[Vo.kind == "none"])
    for lab in BURST:
        row(f"{lab} (todos los segundos)", Vo[Vo.label == lab])
    row("burst (3 severidades, todos los segundos)", Vo[Vo.kind == "burst_trial"])
    # además: segundos de hold sin ninguna señal en NINGUNA variable (límites 0,5-99,5 % de la referencia)
    DET = pd.DataFrame(det)
    DET.to_csv(os.path.join(OUT, "m3c_sil_detector.csv"), index=False)

    # ---------------- separabilidad por variable: AUC de cada variable de telemetría, segundos de hold con tramo vs segundos de ref
    from sklearn.metrics import roc_auc_score
    auc = []
    refs = Vn[Vn.kind == "none"]
    for nm, grp in (("hold, segundos con tramo", Vn[(Vn.kind == "hold_trial") & Vn.en_tramo]),
                    ("hold, todos los segundos", Vn[Vn.kind == "hold_trial"]),
                    ("burst (anterior), todos los segundos", Vo[Vo.kind == "burst_trial"])):
        base = refs if "burst" not in nm else Vo[Vo.kind == "none"]
        for f in FE:
            y = np.r_[np.zeros(len(base)), np.ones(len(grp))]
            x = np.r_[base[f].fillna(0).to_numpy(float), grp[f].fillna(0).to_numpy(float)]
            a = roc_auc_score(y, x) if np.ptp(x) > 0 else np.nan
            auc.append(dict(grupo=nm, variable=f, n_ref=len(base), n_grupo=len(grp), auc=a,
                            media_ref=float(base[f].mean()), media_grupo=float(grp[f].mean())))
    AUC = pd.DataFrame(auc)
    AUC.to_csv(os.path.join(OUT, "m3c_sil_auc.csv"), index=False)

    # ---------------- latencia media por condición (campaña nueva): corrimiento vs. la ref, por sujeto (mediana de corridas)
    lm = subject_med(ex_n.assign(kind=ex_n.label, severity=0.0), ["lat_mean_ms"]).pivot(index="subject", columns="label", values="lat_mean_ms")
    LT = pd.DataFrame({c: dict(media_dif_ms=float((lm[c] - lm["ref"]).mean()), mediana_dif_ms=float((lm[c] - lm["ref"]).median()),
                               min_dif_ms=float((lm[c] - lm["ref"]).min()), max_dif_ms=float((lm[c] - lm["ref"]).max()),
                               sujetos_mayor_que_ref=int((lm[c] > lm["ref"]).sum())) for c in lm.columns if c != "ref"}).T
    LT.index.name = "label"
    LT.to_csv(os.path.join(OUT, "m3c_sil_latencia_condiciones.csv"))
    print(LT.round(4).to_string())

    # ---------------- salida por consola
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 80)
    print("\n== por condición (nueva y anterior) ==")
    print(C.drop(columns=["p_silenciosa_q1", "p_silenciosa_q3"]).round(4).to_string(index=False))
    print("\n== tests (Holm dentro del tipo) ==")
    print(TT[["metric", "kind", "severity", "n", "median_ref", "median_cond", "diff", "friedman_p", "wilcoxon_p", "p_holm", "r"]].round(4).to_string(index=False))
    print("\n== hold vs burst directo ==")
    print(DR.round(4).to_string(index=False))
    print("\n== telemetría: ref vs hold ==")
    print(TI[["metric", "severity", "n", "median_ref", "median_cond", "diff", "friedman_p", "wilcoxon_p", "p_holm", "r"]].round(5).to_string(index=False))
    print("\n== detector por umbrales (LOSO) ==")
    print(DET.round(4).to_string(index=False))
    # medias globales de telemetría por segundo, ref vs hold (descriptivo)
    d = W_n[W_n.kind.isin(["none", "hold_trial"])].groupby("label")[["recv_ratio", "lat_mean", "ia_std", "ia_max", "n_gaps", "exceptions"]].agg(["mean", "max"])
    print("\n== telemetría por segundo (media y máximo) ==")
    print(d.round(5).to_string())


if __name__ == "__main__":
    main()
