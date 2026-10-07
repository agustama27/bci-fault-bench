"""M3 · Tarea 2: exposición de la ventana de decodificación por condición.

Por ensayo (24 por corrida, trials.csv de consumer.py): n_samples = muestras recibidas dentro de la
ventana [onset+2, onset+6] s (1000 si íntegra; 0 si el ensayo fue inválido por <125 muestras).
  tocada     : n_samples < 1000
  mayoritaria: n_samples < 500  (se perdió más de la mitad de la ventana)

Salidas (campaign-2026-09-27/analysis-m3/):
  m3_exposicion_exec.csv        una fila por ejecución
  m3_exposicion_cond.csv        por condición: mediana entre sujetos y global + modelo E + caída de bacc
  m3_condicional.csv            exactitud por ensayo tocado vs. no tocado (CSP+LDA en línea y EEGNet)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:\Users\agustin.tamagusuku\Desktop\bci-fault-bench\campaign-2026-09-27"
OUT = os.path.join(ROOT, "analysis-m3")
W_S = 4.0
LABEL_ORDER = ["ref", "loss-random-0.01", "loss-random-0.05", "loss-random-0.1", "jitter-0.01", "jitter-0.05",
               "jitter-0.1", "delay-0.05", "delay-0.1", "delay-0.25", "disconnect-0.5", "disconnect-1",
               "disconnect-3", "burst_trial-0.1", "burst_trial-0.25", "burst_trial-0.4",
               "disconnect_trial-0.5", "disconnect_trial-1", "disconnect_trial-2"]


def main():
    execs = pd.read_csv(os.path.join(ROOT, "analysis", "execs.csv"))
    rows, cond_rows = [], []
    for _, e in execs.iterrows():
        d = os.path.join(ROOT, "raw", e.exec_id)
        tr = pd.read_csv(os.path.join(d, "trials.csv"))
        te = pd.read_csv(os.path.join(d, "trials_eegnet.csv"))
        assert len(tr) == len(te) == 24, (e.exec_id, len(tr), len(te))
        assert np.allclose(tr.onset, te.onset)
        n = tr.n_samples.to_numpy()
        rows.append(dict(exec_id=e.exec_id, subject=e.subject, run=e.run, kind=e.kind, severity=e.severity,
                         label=e.label, n_trials=len(tr), outages=e.outages, secs_total=e.secs_total,
                         k_lt1000=int((n < 1000).sum()), k_lt500=int((n < 500).sum()),
                         k_invalid=int((tr.valid == 0).sum()),
                         p_lt1000=float((n < 1000).mean()), p_lt500=float((n < 500).mean()),
                         lost_frac_mean=float((1 - n / 1000).mean()),
                         n_med=float(np.median(n)), n_min=int(n.min())))
        # exactitud por ensayo, tocado vs no tocado (solo válidos)
        for dec, df in (("CSP+LDA", tr), ("EEGNet", te)):
            v = (df.valid == 1) & (df.pred.notna())
            ok = (df.pred[v].astype(float) == df.label[v].astype(float)).to_numpy()
            touched = (n[v.to_numpy()] < 1000)
            cond_rows.append(dict(label=e.label, kind=e.kind, severity=e.severity, decoder=dec, subject=e.subject,
                                  n_t=int(touched.sum()), ok_t=int(ok[touched].sum()),
                                  n_u=int((~touched).sum()), ok_u=int(ok[~touched].sum())))
    X = pd.DataFrame(rows)
    X.to_csv(os.path.join(OUT, "m3_exposicion_exec.csv"), index=False)

    # ---------- por condición
    out = []
    for lab in LABEL_ORDER:
        g = X[X.label == lab]
        subj = g.groupby("subject")[["p_lt1000", "p_lt500", "lost_frac_mean"]].median()
        out.append(dict(
            label=lab, kind=g.kind.iloc[0], severity=g.severity.iloc[0], n_exec=len(g),
            med_subj_lt1000=subj.p_lt1000.median(), q1_subj_lt1000=subj.p_lt1000.quantile(.25), q3_subj_lt1000=subj.p_lt1000.quantile(.75),
            glob_lt1000=g.k_lt1000.sum() / g.n_trials.sum(),
            med_subj_lt500=subj.p_lt500.median(), glob_lt500=g.k_lt500.sum() / g.n_trials.sum(),
            glob_invalid=g.k_invalid.sum() / g.n_trials.sum(),
            lost_frac_med_subj=subj.lost_frac_mean.median(), lost_frac_glob=g.lost_frac_mean.mean(),
            n_med_glob=float(g.n_med.median()),
        ))
    C = pd.DataFrame(out)

    # ---------- modelo E = N (d + r + w) / T para desconexión uniforme
    T1 = pd.read_csv(os.path.join(OUT, "m3_cortes.csv"))
    hueco = T1[T1.kind == "disconnect"].groupby("severity").hueco_ts.median()   # d + r observado (tarea 1)
    for sev, h in hueco.items():
        lab = f"disconnect-{sev:g}"
        g = X[X.label == lab]
        N = g.outages.to_numpy(float)
        T = g.secs_total.to_numpy(float)
        e_nom = (N * (sev + W_S) / T)                # r = 0
        e_mod = (N * (h + W_S) / T)                  # d + r con r = hueco - d de la tarea 1
        r = h - sev
        i = C.index[C.label == lab][0]
        # refinamiento (aporte propio): la misma esperanza pero con la zona de guarda (cortes solo en
        # [15, T-15-d]) y los inicios de ventana reales de cada corrida; h = d + r de la tarea 1.
        ex = []
        for _, e2 in g.iterrows():
            on = pd.read_csv(os.path.join(ROOT, "raw", e2.exec_id, "trials.csv")).onset.to_numpy(float)
            a = on + 2.0
            lo_s, hi_s = 15.0, e2.secs_total - 15.0 - sev
            ov = np.clip(np.minimum(a + W_S, hi_s) - np.maximum(a - h, lo_s), 0, None)
            ex.append(e2.outages * ov.mean() / (hi_s - lo_s))
        C.loc[i, "E_guarda"] = np.median(ex)
        C.loc[i, "obs_mean_exec"] = g.p_lt1000.mean()
        C.loc[i, "obs_se_exec"] = g.p_lt1000.std(ddof=1) / np.sqrt(len(g))
        C.loc[i, "E_guarda_mean"] = float(np.mean(ex))
        C.loc[i, "E_r0"] = np.median(e_nom)
        C.loc[i, "E_modelo"] = np.median(e_mod)
        C.loc[i, "r_tarea1"] = r
        C.loc[i, "hueco_tarea1"] = h
        # ocupación por corte: ensayos tocados por corte (observado) vs. modelo (h + w)/ (T/24)
        C.loc[i, "obs_trials_por_corte"] = (g.k_lt1000.sum() / g.outages.sum())
        C.loc[i, "mod_trials_por_corte"] = np.median((h + W_S) / (T / 24.0))
    # disconnect_trial: corte al inicio de la ventana, d + r a partir de allí
    hueco_t = T1[T1.kind == "disconnect_trial"].groupby("severity").hueco_ts.median()
    for sev, h in hueco_t.items():
        i = C.index[C.label == f"disconnect_trial-{sev:g}"][0]
        C.loc[i, "hueco_tarea1"] = h
        C.loc[i, "r_tarea1"] = h - sev
        C.loc[i, "n_esperado_ventana"] = max(0.0, 1000 - h * 250)   # muestras restantes si el corte arranca en 2 s

    # ---------- caída de balanced accuracy (por sujeto: condición - referencia)
    for dec, folder in (("CSP+LDA", "analysis"), ("EEGNet", "analysis-eegnet")):
        med = pd.read_csv(os.path.join(ROOT, folder, "medianas_por_sujeto.csv"))
        ref = med[med.label == "ref"].set_index("subject").bacc
        t = pd.read_csv(os.path.join(ROOT, folder, "t_desempeno.csv"))
        for lab in LABEL_ORDER[1:]:
            s = med[med.label == lab].set_index("subject").bacc
            dd = (s - ref).dropna()
            i = C.index[C.label == lab][0]
            C.loc[i, f"dBA_mean_{dec}"] = dd.mean()
            C.loc[i, f"dBA_med_{dec}"] = dd.median()
            tt = t[(t.kind == C.loc[i, "kind"]) & (np.isclose(t.severity, C.loc[i, "severity"]))]
            C.loc[i, f"p_holm_{dec}"] = float(tt.p_holm.iloc[0])
            C.loc[i, f"wilcoxon_p_{dec}"] = float(tt.wilcoxon_p.iloc[0])
        C.loc[C.label == "ref", f"bacc_ref_{dec}"] = ref.median()
    C.to_csv(os.path.join(OUT, "m3_exposicion_cond.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(C[["label", "n_exec", "med_subj_lt1000", "glob_lt1000", "med_subj_lt500", "glob_lt500", "glob_invalid",
             "lost_frac_med_subj", "lost_frac_glob", "n_med_glob"]].round(4).to_string())
    print(C[["label", "hueco_tarea1", "r_tarea1", "E_r0", "E_modelo", "E_guarda", "med_subj_lt1000", "glob_lt1000",
             "obs_trials_por_corte", "mod_trials_por_corte", "n_esperado_ventana"]].dropna(subset=["hueco_tarea1"]).round(4).to_string())
    print(C[["label", "dBA_mean_CSP+LDA", "dBA_med_CSP+LDA", "p_holm_CSP+LDA", "dBA_mean_EEGNet", "dBA_med_EEGNet",
             "p_holm_EEGNet"]].round(4).to_string())

    # ---------- relación exposición - caída (n=18 condiciones)
    D = C[C.label != "ref"].copy()
    print("\nSpearman entre condiciones (n=18):")
    rel = []
    for dec in ("CSP+LDA", "EEGNet"):
        for xname in ("med_subj_lt1000", "glob_lt500", "lost_frac_glob"):
            r = stats.spearmanr(D[xname], D[f"dBA_mean_{dec}"])
            # regresión por origen: dBA = b * x  (mínimos cuadrados sin intercepto)
            x = D[xname].to_numpy(float); y = D[f"dBA_mean_{dec}"].to_numpy(float)
            b = float((x * y).sum() / (x * x).sum())
            rel.append(dict(decoder=dec, x=xname, rho=r.statistic, p=r.pvalue, slope_origin=b))
        # restringida a las condiciones con exposición plena (ventana tocada >=50 %)
    R = pd.DataFrame(rel)
    R.to_csv(os.path.join(OUT, "m3_exposicion_vs_caida.csv"), index=False)
    print(R.round(4).to_string())
    print("\nSubconjunto con lost_frac_glob>0.05 (condiciones con dosis apreciable):")
    D2 = D[D.lost_frac_glob > 0.05]
    for dec in ("CSP+LDA", "EEGNet"):
        r = stats.spearmanr(D2.lost_frac_glob, D2[f"dBA_mean_{dec}"])
        print(dec, len(D2), "rho=", round(r.statistic, 3), "p=", round(r.pvalue, 4))
        print(D2[["label", "lost_frac_glob", f"dBA_mean_{dec}"]].round(4).to_string(index=False))

    # ---------- exactitud tocado vs no tocado
    K = pd.DataFrame(cond_rows)
    A = (K.groupby(["label", "decoder"])[["n_t", "ok_t", "n_u", "ok_u"]].sum().reset_index())
    A["acc_tocado"] = A.ok_t / A.n_t.replace(0, np.nan)
    A["acc_no_tocado"] = A.ok_u / A.n_u.replace(0, np.nan)
    A["delta"] = A.acc_tocado - A.acc_no_tocado
    # IC de Wilson 95 % para cada proporción
    def wilson(k, n):
        if n == 0:
            return (np.nan, np.nan)
        z = 1.96; p = k / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return (c - h, c + h)
    for col, kk, nn in (("t", "ok_t", "n_t"), ("u", "ok_u", "n_u")):
        ci = [wilson(a, b) for a, b in zip(A[kk], A[nn])]
        A[f"lo_{col}"] = [c[0] for c in ci]; A[f"hi_{col}"] = [c[1] for c in ci]
    A["_o"] = A.label.map({l: i for i, l in enumerate(LABEL_ORDER)})
    A = A.sort_values(["decoder", "_o"]).drop(columns="_o")
    A.to_csv(os.path.join(OUT, "m3_condicional.csv"), index=False)
    print("\nExactitud por ensayo tocado vs no tocado (agrupado, ensayos válidos)")
    print(A[A.label.isin(["ref", "disconnect-0.5", "disconnect-1", "disconnect-3", "loss-random-0.1",
                          "burst_trial-0.1", "burst_trial-0.25", "burst_trial-0.4"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
