"""M3b · Tarea B: posición del hueco dentro de la ventana en burst_trial. EXPLORATORIO, post hoc.

Recuperación de la posición (no hay segmentos guardados ni fault_log útil: fault_log.jsonl de burst_trial está vacío):
  1. El inyector (injector.py) sortea, para CADA evento de la corrida (48 marcadores, no solo los 24 decodificados),
     un desplazamiento u = rng.integers(0, 1000 - L + 1) con rng = default_rng(seed); la semilla está en producer.json
     (fault.seed). Reproducir esa secuencia da u para cada evento: el hueco ocupa las muestras
     [onset·250 + 500 + u, + L) de la corrida (500 = 2 s de TMIN).
  2. Los 24 ensayos decodificados son los eventos de clase 1 y 2; su índice entre los 48 eventos se reconstruye con los
     intervalos entre onsets (≈ 8,0 s por evento): k_i = k_{i-1} + round(Δonset / 8,0).
  3. VALIDACIÓN INDEPENDIENTE contra telemetry.csv: el déficit de muestras por segundo (250 − n_samples) en los
     segundos de la ventana debe coincidir EXACTAMENTE con el que predice (u, L). Solo se usan los ensayos que validan.

Posición: u_frac = u / (1000 − L) ∈ [0, 1] (0 = el hueco arranca al inicio de la ventana, 1 = termina al final).
  inicio / medio / fin = tercios de u_frac (primario, equiprobables por construcción).
  Secundaria: centro del hueco en tercios de la ventana.

Salidas (analysis-m3b/): m3b_B_trials.csv, m3b_B_posicion.csv, m3b_B_subject.csv, m3b_B_pruebas.csv, m3b_B_validacion.csv
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from m3b_common import OUT, ROOT, load_paired, wilcoxon_paired

SF = 250
WIN = 1000
TMIN_S = 500
EV_PERIOD = 8.0
POS = ["inicio", "medio", "fin"]


def recover(exec_id: str):
    d = os.path.join(ROOT, "raw", exec_id)
    p = json.load(open(os.path.join(d, "producer.json")))
    tr = pd.read_csv(os.path.join(d, "trials.csv"))
    te = pd.read_csv(os.path.join(d, "telemetry.csv"))
    L = int(round(p["fault"]["severity"] * WIN))
    n_ev = int(p["n_events"])
    rng = np.random.default_rng(int(p["fault"]["seed"]))
    u = np.array([int(rng.integers(0, max(WIN - L, 0) + 1)) for _ in range(n_ev)])
    on = tr.onset.to_numpy()
    k_rel = np.concatenate([[0], np.cumsum(np.round(np.diff(on) / EV_PERIOD).astype(int))])
    best = None
    for k0 in range(0, max(n_ev - int(k_rel[-1]), 1)):          # índice del primer decodificado (normalmente 0)
        k = k_rel + k0
        if k[-1] >= n_ev:
            break
        match = []
        for i in range(len(on)):
            s = int(round(on[i] * SF)) + TMIN_S + int(u[k[i]]); e = s + L
            pred = {sec: min(e, (sec + 1) * SF) - max(s, sec * SF) for sec in range(s // SF, (e - 1) // SF + 1)}
            obs = {}
            for sec in range(int(on[i]) , int(on[i]) + 8):
                if 0 <= sec < len(te):
                    dfc = SF - int(te.n_samples.iloc[sec])
                    if dfc != 0:
                        obs[sec] = dfc
            # solo los segundos dentro de la ventana de este ensayo (±1 s de margen para el filtro no importa: no hay fallo)
            lo, hi = (int(on[i]) + 2) // 1, int(on[i]) + 6
            obs = {a: b for a, b in obs.items() if lo <= a <= hi}
            match.append(pred == obs)
        score = int(np.sum(match))
        if best is None or score > best[0]:
            best = (score, k0, k, np.array(match))
    score, k0, k, match = best
    return dict(exec_id=exec_id, L=L, u=u[k], k=k, k0=k0, match=match, n_match=score)


def main():
    D = load_paired()
    Bx = D[D.kind == "burst_trial"].copy()
    ids = sorted(Bx.exec_id.unique())
    recs = {}
    val_rows = []
    for eid in ids:
        r = recover(eid)
        recs[eid] = r
        val_rows.append(dict(exec_id=eid, severity=r["L"] / WIN, n_trials=len(r["k"]), n_match=r["n_match"], k0=r["k0"]))
    V = pd.DataFrame(val_rows)
    V["pct"] = V.n_match / V.n_trials
    V.to_csv(os.path.join(OUT, "m3b_B_validacion.csv"), index=False)
    print("Validación contra telemetría (déficit por segundo exacto):")
    print(V.groupby("severity").agg(execs=("exec_id", "size"), trials=("n_trials", "sum"), matched=("n_match", "sum"),
                                    execs_all_match=("pct", lambda s: int((s == 1).sum()))).to_string())
    print("k0 != 0 en", int((V.k0 != 0).sum()), "ejecuciones")

    # una fila por ensayo con posición validada
    rows = []
    for eid in ids:
        r = recs[eid]
        for i in range(len(r["k"])):
            rows.append(dict(exec_id=eid, i=i, L=r["L"], u=int(r["u"][i]), ev_index=int(r["k"][i]), valida_tele=bool(r["match"][i])))
    P = pd.DataFrame(rows)
    T = Bx.merge(P, on=["exec_id", "i"], how="left")
    assert T.valida_tele.notna().all()
    T["u_frac"] = T.u / (WIN - T.L)
    T["pos"] = pd.cut(T.u_frac, [-0.001, 1 / 3, 2 / 3, 1.001], labels=POS)
    T["centro"] = T.u + T.L / 2
    T["pos_centro"] = pd.cut(T.centro, [0, WIN / 3, 2 * WIN / 3, WIN + 1], labels=POS)
    T["quint"] = pd.cut(T.u_frac, [-0.001, .2, .4, .6, .8, 1.001], labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
    for tag in ("csp", "net"):
        T[f"harm_{tag}"] = ((T[f"ref_ok_{tag}"] == 1) & (T[f"ok_{tag}"] == 0)).astype(int)
        T[f"help_{tag}"] = ((T[f"ref_ok_{tag}"] == 0) & (T[f"ok_{tag}"] == 1)).astype(int)
    print("\nensayos:", len(T), "| validados por telemetría:", int(T.valida_tele.sum()))
    T.to_csv(os.path.join(OUT, "m3b_B_trials.csv"), index=False)
    T = T[T.valida_tele].copy()

    # ---------------- por severidad × posición (agrupado)
    res, subj_rows = [], []
    for dec, tag in (("CSP+LDA", "csp"), ("EEGNet", "net")):
        for sev, g in list(T.groupby("severity")) + [("todas", T)]:
            for grp_col in ("pos", "pos_centro"):
                for pos in POS:
                    h = g[g[grp_col] == pos]
                    res.append(dict(decoder=dec, severity=sev, clasif=grp_col, pos=pos, n=len(h),
                                    n_subj=h.subject.nunique(),
                                    chg=h[f"chg_{tag}"].mean(), harm=h[f"harm_{tag}"].mean(), help=h[f"help_{tag}"].mean(),
                                    acc=h[f"ok_{tag}"].mean(), acc_ref_mismos=h[f"ref_ok_{tag}"].mean(),
                                    d_acc=h[f"ok_{tag}"].mean() - h[f"ref_ok_{tag}"].mean()))
        # por sujeto (primario: pos)
        s = (T.groupby(["severity", "subject", "pos"], observed=True)
               .agg(n=("i", "size"), chg=(f"chg_{tag}", "mean"), acc=(f"ok_{tag}", "mean"), acc_ref=(f"ref_ok_{tag}", "mean"))
               .reset_index())
        s.insert(0, "decoder", dec)
        subj_rows.append(s)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "m3b_B_posicion.csv"), index=False)
    S = pd.concat(subj_rows, ignore_index=True)
    S["dacc"] = S.acc - S.acc_ref        # acierto menos el de la referencia en esos mismos ensayos
    S.to_csv(os.path.join(OUT, "m3b_B_subject.csv"), index=False)

    # ---------------- pruebas a nivel sujeto
    tests = []
    for dec, tag in (("CSP+LDA", "csp"), ("EEGNet", "net")):
        s = S[S.decoder == dec]
        # mean sobre severidades (cada sujeto × posición promedia las 3 severidades): ajusta por severidad
        s_all = s.groupby(["subject", "pos"], observed=True)[["chg", "acc", "acc_ref", "dacc"]].mean().reset_index()
        s_all["severity"] = "todas (prom.)"
        for sev, g in list(s.groupby("severity")) + [("todas (prom.)", s_all)]:
            for metric in ("chg", "acc", "dacc"):
                w = g.pivot(index="subject", columns="pos", values=metric)[POS].dropna()
                if len(w) < 4:
                    continue
                fr = stats.friedmanchisquare(*[w[c].to_numpy() for c in POS])
                _, _, p_if = wilcoxon_paired(w["inicio"].to_numpy(), w["fin"].to_numpy())
                _, _, p_im = wilcoxon_paired(w["inicio"].to_numpy(), w["medio"].to_numpy())
                _, _, p_mf = wilcoxon_paired(w["medio"].to_numpy(), w["fin"].to_numpy())
                tests.append(dict(decoder=dec, severity=sev, metric=metric, n_subj=len(w),
                                  med_inicio=w["inicio"].median(), med_medio=w["medio"].median(), med_fin=w["fin"].median(),
                                  friedman_chi2=fr.statistic, friedman_p=fr.pvalue,
                                  p_inicio_vs_fin=p_if, p_inicio_vs_medio=p_im, p_medio_vs_fin=p_mf))
        # correlación a nivel ensayo (descriptiva): posición continua vs cambio, dentro de severidad
        for sev, g in list(T.groupby("severity")) + [("todas", T)]:
            for xname in ("u_frac",):
                rr = stats.spearmanr(g[xname], g[f"chg_{tag}"])
                tests.append(dict(decoder=dec, severity=sev, metric=f"spearman_{xname}_vs_chg(ensayo)", n_subj=len(g),
                                  med_inicio=rr.statistic, friedman_p=rr.pvalue))
                rr = stats.spearmanr(g[xname], g[f"ok_{tag}"])
                tests.append(dict(decoder=dec, severity=sev, metric=f"spearman_{xname}_vs_acc(ensayo)", n_subj=len(g),
                                  med_inicio=rr.statistic, friedman_p=rr.pvalue))
    Pt = pd.DataFrame(tests)
    Pt.to_csv(os.path.join(OUT, "m3b_B_pruebas.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
    print("\n== severidad × posición (primario: tercios de u_frac)")
    print(R[R.clasif == "pos"].drop(columns="clasif").round(3).to_string(index=False))
    print("\n== pruebas")
    print(Pt.round(4).to_string(index=False))
    print("\n== sensibilidad: clasificación por centro del hueco")
    print(R[(R.clasif == "pos_centro") & (R.severity != "todas")].drop(columns="clasif").round(3).to_string(index=False))
    # quintiles
    q = []
    for dec, tag in (("CSP+LDA", "csp"), ("EEGNet", "net")):
        for sev, g in T.groupby("severity"):
            for qq, h in g.groupby("quint", observed=True):
                q.append(dict(decoder=dec, severity=sev, u_frac_pct=qq, n=len(h), chg=h[f"chg_{tag}"].mean(), acc=h[f"ok_{tag}"].mean()))
    Q = pd.DataFrame(q); Q.to_csv(os.path.join(OUT, "m3b_B_quintiles.csv"), index=False)
    print("\n== quintiles de posición")
    print(Q.pivot_table(index=["decoder", "severity"], columns="u_frac_pct", values="chg").round(3).to_string())


if __name__ == "__main__":
    main()
