"""M3c · Tablas en Markdown para la nota de resultados (lee los CSV de campaign-m3/analysis/, no recalcula nada).

Salida: campaign-m3/analysis/m3c_tablas.md (un bloque '<!-- T:nombre -->' por tabla) y un diccionario TABLAS que usa
m3c_nota.py para armar la nota. Puntos decimales en las tablas.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from m3c_common import OUT

rd = lambda f: pd.read_csv(os.path.join(OUT, f))


def md(df, fmt=".3f"):
    """DataFrame -> tabla Markdown (sin depender de tabulate); floats con `fmt`."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "n.d." if np.isnan(v) else format(v, fmt)
        return str(v)
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(cell(v) for v in r.tolist()) + " |")
    return "\n".join(out)


def pv(p):
    return "n.d." if pd.isna(p) else (f"{p:.4f}" if p < 0.001 else f"{p:.3f}")


def pct(x, d=1):
    return "n.d." if pd.isna(x) else f"{100 * x:.{d}f}"


def build():
    T = {}
    # ---------------------------------------------------------------- 1. reconexión
    R, ST = rd("m3c_espera_resumen.csv"), rd("m3c_modelo_escalonado.csv")
    G = rd("m3c_regla.csv")
    rows = []
    for _, r in R.iterrows():
        lin = ST[(ST.campania == "campaña nueva") & (ST.condicion == r.label) & ST.modelo.str.startswith("lineal")].iloc[0]
        esc = ST[(ST.campania == "campaña nueva") & (ST.condicion == r.label) & ST.modelo.str.startswith("escalonado")].iloc[0]
        gi = G[(G.label == r.label) & (G.hueco == "hueco_ia_corr") & (G.d_usado == "dnom")].iloc[0]
        rows.append({"Condición": r.label, "d nominal (s)": r.d_plan, "Cortes": int(r.n_cortes),
                     "Hueco (s) mediana [mín–máx]": f"{r.hueco_ts_med:.2f} [{r.hueco_ts_min:.2f}–{r.hueco_ts_max:.2f}]",
                     "Hueco ia_max (s) mediana": f"{r.hueco_ia_med:.3f}",
                     "Espera = hueco − d (s) mediana [mín–máx]": f"{r.espera_ts_med:.2f} [{r.espera_ts_min:.2f}–{r.espera_ts_max:.2f}]",
                     "Regla lineal: predicción (s)": f"{lin.pred:.2f}",
                     "Lineal: dentro de ±40 ms": f"{int(round(lin.pct_dentro_40ms * r.n_cortes / 100))}/{int(r.n_cortes)}",
                     "Escalonada: predicción (s)": f"{esc.pred:.2f}",
                     "Escalonada: dentro de ±40 ms": f"{int(round(esc.pct_dentro_40ms * r.n_cortes / 100))}/{int(r.n_cortes)}"})
    T["recon"] = md(pd.DataFrame(rows))
    sel = ST[(ST.condicion == "TODAS")].copy()
    sel["Modelo"] = sel.modelo.str.replace("max(1,52; ", "max(1,52; ", regex=False)
    T["recon_global"] = md(sel.assign(**{"Cortes": sel.n.astype(int), "Campaña": sel.campania, "MAE (ms)": sel.mae_ms,
                                         "Error máx. (ms)": sel.max_abs_ms, "Dentro de ±40 ms (%)": sel.pct_dentro_40ms})
                           [["Campaña", "Modelo", "Cortes", "MAE (ms)", "Error máx. (ms)", "Dentro de ±40 ms (%)"]], ".2f")
    # medida alternativa del hueco (ia_max) en los cortes de 1 s
    g2 = G[(G.d_usado == "dnom") & (G.label != "TODAS")]
    t = g2.pivot(index="label", columns="hueco", values="pct_dentro_40ms")
    T["recon_ia"] = md(t.reset_index().rename(columns={"label": "Condición", "hueco_ts": "hueco_ts (gap_s)", "hueco_ia": "ia_max sin corregir",
                                                       "hueco_ia_corr": "ia_max − 0,04 s"}), ".1f")

    # ---------------------------------------------------------------- 2. exposición
    C, TE = rd("m3c_exp_cond.csv"), rd("m3c_exp_tests.csv")
    rows = []
    for _, r in C.iterrows():
        rows.append({"Condición": r.label, "K": int(r.K),
                     "Tocados, mediana entre sujetos [Q1–Q3] (%)": f"{pct(r.p_touched_med_subj)} [{pct(r.p_touched_q1)}–{pct(r.p_touched_q3)}]",
                     "Tocados, global (%) (k/n)": f"{pct(r.p_touched_glob)} ({int(r.k_touched_total)}/{int(r.n_trials_total)})",
                     "Modelo K(1,52+4)/387 (%)": pct(r.E_modelo), "Ensayos inválidos": int(r.k_invalid_total),
                     "Muestras perdidas (% de la ventana, global)": pct(abs(r.lost_frac_glob), 2)})
    T["exp_expo"] = md(pd.DataFrame(rows))
    rows = []
    for dec, c in (("CSP+LDA", "csp"), ("EEGNet", "net")):
        for _, r in C.iterrows():
            if r.K == 0:
                rows.append({"Decodificador": dec, "Condición": "ref", "BA mediana": f"{r[f'bacc_{c}_med']:.4f}", "Δ vs ref, mediana": "", "Δ vs ref, media": "",
                             "Sujetos peor/igual/mejor": "", "p Wilcoxon": "", "p Holm": "", "r": "", "Friedman p": ""})
                continue
            nb, nm = int(r[f"n_subj_peor_{c}"]), int(r[f"n_subj_mejor_{c}"])
            rows.append({"Decodificador": dec, "Condición": r.label, "BA mediana": f"{r[f'bacc_{c}_med']:.4f}",
                         "Δ vs ref, mediana": f"{r[f'dBA_med_{c}']:+.4f}", "Δ vs ref, media": f"{r[f'dBA_mean_{c}']:+.4f}",
                         "Sujetos peor/igual/mejor": f"{nb}/{9 - nb - nm}/{nm}", "p Wilcoxon": pv(r[f"wilcoxon_p_{c}"]),
                         "p Holm": pv(r[f"p_holm_{c}"]), "r": "n.d." if pd.isna(r[f"r_{c}"]) else f"{r[f'r_{c}']:.2f}",
                         "Friedman p": pv(r[f"friedman_p_{c}"])})
    T["exp_ba"] = md(pd.DataFrame(rows))
    X = rd("m3c_exp_cambio.csv")
    X = X[X.label.isin(["disconnect-n2-1", "disconnect-n5-1", "disconnect-n10-1", "disconnect-n20-1", "TODAS"])]
    X = X.assign(**{"Cambio (%)": 100 * X.tasa_cambio, "IC95 Wilson (%)": X.apply(lambda r: f"{100 * r.ic_lo:.1f}–{100 * r.ic_hi:.1f}", axis=1),
                    "Acierto cond. (%)": 100 * X.acc_cond, "Acierto ref. (%)": 100 * X.acc_ref})
    T["exp_cambio"] = md(X[["label", "decoder", "grupo", "n_ensayos", "n_cambios", "Cambio (%)", "IC95 Wilson (%)", "n_perdidas", "n_ganadas",
                            "Acierto cond. (%)", "Acierto ref. (%)"]].rename(columns={"label": "Condición", "decoder": "Decodificador", "grupo": "Grupo",
                                                                                         "n_ensayos": "Ensayos", "n_cambios": "Cambios", "n_perdidas": "Acertaba→falla",
                                                                                         "n_ganadas": "Fallaba→acierta"}), ".1f")
    Dd = rd("m3c_exp_dosis.csv")
    Dd = Dd.assign(**{"Cambio (%)": 100 * Dd.tasa_cambio, "IC95 (%)": Dd.apply(lambda r: f"{100 * r.ic_lo:.1f}–{100 * r.ic_hi:.1f}", axis=1),
                      "Acierto cond. (%)": 100 * Dd.acc_cond, "Acierto ref. (%)": 100 * Dd.acc_ref})
    T["exp_dosis"] = md(Dd[["bin", "decoder", "n_ensayos", "n_cambios", "Cambio (%)", "IC95 (%)", "n_perdidas", "n_ganadas", "Acierto cond. (%)",
                            "Acierto ref. (%)"]].rename(columns={"bin": "Muestras en la ventana", "decoder": "Decodificador", "n_ensayos": "Ensayos",
                                                                "n_cambios": "Cambios", "n_perdidas": "Acertaba→falla", "n_ganadas": "Fallaba→acierta"}), ".1f")

    # ---------------------------------------------------------------- 3. retención vs borrado
    Cc = rd("m3c_ret_cond.csv")
    rows = []
    for _, r in Cc.iterrows():
        rows.append({"Familia": r.familia, "Decodificador": r.decoder, "Severidad": f"{int(round(r.severidad * 100))} %", "BA ref (mediana)": f"{r.bacc_ref_med:.4f}",
                     "BA cond. (mediana)": f"{r.bacc_cond_med:.4f}", "Δ mediana": f"{r.dBA_med:+.4f}", "Δ media": f"{r.dBA_mean:+.4f}",
                     "Sujetos peor/igual/mejor": f"{int(r.n_peor)}/{int(r.n_igual)}/{int(r.n_mejor)}", "p Wilcoxon": pv(r.wilcoxon_p), "p Holm": pv(r.p_holm),
                     "r": f"{r.r:.2f}", "Friedman p": pv(r.friedman_p)})
    T["ret_ba"] = md(pd.DataFrame(rows))
    DR = rd("m3c_ret_directo.csv")
    T["ret_directo"] = md(DR.assign(**{"Severidad": DR.severidad.map(lambda s: f"{int(round(s * 100))} %"), "p Wilcoxon": DR.wilcoxon_p.map(pv),
                                      "p Holm": DR.p_holm.map(pv), "r": DR.r.map(lambda x: f"{x:.2f}"),
                                      "Sujetos con menos caída en hold / más": DR.apply(lambda r: f"{int(r.n_hold_menos_caida)}/{int(r.n_hold_mas_caida)}", axis=1)})
                          [["decoder", "Severidad", "n", "dBA_hold_med", "dBA_burst_med", "dif_hold_menos_burst_med", "p Wilcoxon", "p Holm", "r",
                            "Sujetos con menos caída en hold / más"]]
                          .rename(columns={"decoder": "Decodificador", "dBA_hold_med": "Δ hold (mediana)", "dBA_burst_med": "Δ burst (mediana)",
                                           "dif_hold_menos_burst_med": "Δhold − Δburst (mediana)"}), ".4f")
    X = rd("m3c_ret_cambio.csv")
    X = X[X.severidad != "todas"].copy()
    X["sev"] = X.severidad.astype(float).map(lambda s: f"{int(round(s * 100))} %")
    X = X.assign(**{"Cambio (%)": 100 * X.tasa_cambio, "IC95 (%)": X.apply(lambda r: f"{100 * r.ic_lo:.1f}–{100 * r.ic_hi:.1f}", axis=1),
                    "Acierto cond. (%)": 100 * X.acc_cond, "Acierto ref. (%)": 100 * X.acc_ref})
    T["ret_cambio"] = md(X[["familia", "decoder", "sev", "grupo", "n_ensayos", "n_cambios", "Cambio (%)", "IC95 (%)", "n_perdidas", "n_ganadas",
                            "Acierto cond. (%)", "Acierto ref. (%)"]].rename(columns={"familia": "Familia", "decoder": "Decodificador", "sev": "Severidad",
                                                                                       "grupo": "Grupo", "n_ensayos": "Ensayos", "n_cambios": "Cambios",
                                                                                       "n_perdidas": "Acertaba→falla", "n_ganadas": "Fallaba→acierta"}), ".1f")
    SR = rd("m3c_ret_cambio_directo.csv")
    T["ret_cambio_sujeto"] = md(SR.assign(**{"Severidad": SR.severidad.map(lambda s: f"{int(round(s * 100))} %"), "p Wilcoxon": SR.wilcoxon_p.map(pv),
                                            "p Holm": SR.p_holm.map(pv), "r": SR.r.map(lambda x: f"{x:.2f}")})
                                [["decoder", "Severidad", "n", "cambio_hold_med_subj", "cambio_burst_med_subj", "dif_med", "p Wilcoxon", "p Holm", "r"]]
                                .rename(columns={"decoder": "Decodificador", "cambio_hold_med_subj": "Cambio hold (mediana entre sujetos)",
                                                 "cambio_burst_med_subj": "Cambio burst (mediana entre sujetos)", "dif_med": "Dif. (mediana)"}), ".4f")

    # ---------------------------------------------------------------- 4. fallas silenciosas
    C = rd("m3c_sil_cond.csv")
    C = C[C.campania.isin(["nueva", "anterior"])]
    keep = ["ref", "hold_trial-0.1", "hold_trial-0.25", "hold_trial-0.4", "burst_trial-0.1", "burst_trial-0.25", "burst_trial-0.4"]
    rows = []
    for _, r in C[C.label.isin(keep)].iterrows():
        rows.append({"Campaña": r.campania, "Condición": r.label, "Evaluables (s)": int(r.secs_eval_total), "Degradados (s)": int(r.secs_degradados_total),
                     "Operativos y degradados (s)": int(r.secs_silenciosos_total),
                     "Proporción por ejecución, mediana entre sujetos": f"{r.p_silenciosa_med_subj:.4f}",
                     "Máx. entre sujetos": f"{r.p_silenciosa_max_subj:.4f}", "Global (%)": f"{100 * r.p_silenciosa_glob:.2f}",
                     "Degradado que es silencioso (%)": "n.d." if pd.isna(r.frac_degradado_que_es_silencioso) else f"{100 * r.frac_degradado_que_es_silencioso:.1f}",
                     "Ejecuciones con ≥1 s (de 45)": int(r.ejec_con_silenciosa), "Sujetos con ≥1 s (de 9)": f"{int(r.subj_con_silenciosa)} {r.subj_con_silenciosa_lista}",
                     "Segundos sin muestras": int(r.secs_sin_muestras_total)})
    T["sil_cond"] = md(pd.DataFrame(rows))
    rows = []
    for _, r in C[~C.label.isin(keep)].iterrows():
        rows.append({"Condición": r.label, "Evaluables (s)": int(r.secs_eval_total), "Degradados (s)": int(r.secs_degradados_total),
                     "Operativos y degradados (s)": int(r.secs_silenciosos_total), "Mediana entre sujetos": f"{r.p_silenciosa_med_subj:.4f}",
                     "Ejecuciones con ≥1 s": int(r.ejec_con_silenciosa), "Segundos sin muestras": int(r.secs_sin_muestras_total)})
    T["sil_otras"] = md(pd.DataFrame(rows))
    TI = rd("m3c_sil_telemetria.csv")
    TI = TI[TI.metric.isin(["lat_mean_ms", "lat_max_ms", "ia_std_ms", "ia_max_ms", "recv_ratio", "n_gaps", "gap_s", "secs_no_data"])]
    T["sil_tele"] = md(TI.assign(**{"Severidad": TI.severity.map(lambda s: f"{int(round(s * 100))} %"), "p Wilcoxon": TI.wilcoxon_p.map(pv), "p Holm": TI.p_holm.map(pv),
                                    "r": TI.r.map(lambda x: "n.d." if pd.isna(x) else f"{x:.2f}")})
                       [["metric", "Severidad", "median_ref", "median_cond", "diff", "p Wilcoxon", "p Holm", "r"]]
                       .rename(columns={"metric": "Variable", "median_ref": "Ref (mediana entre sujetos)", "median_cond": "Hold (mediana entre sujetos)", "diff": "Diferencia"}), ".5f")
    DET = rd("m3c_sil_detector.csv")
    DET = DET[DET.grupo.str.contains("ref|\\(3 severidades|segundos con tramo\\)|todos los segundos", regex=True)]
    T["sil_det"] = md(DET.assign(**{"Tasa de alarma (%)": 100 * DET.tasa_alarma, "Recall sobre degradados (%)": 100 * DET.recall_degradados})
                      [["grupo", "n_seg", "n_alarmas", "Tasa de alarma (%)", "n_degradados", "alarmas_en_degradados", "Recall sobre degradados (%)",
                        "n_silenciosos", "alarmas_en_silenciosos"]]
                      .rename(columns={"grupo": "Grupo de segundos", "n_seg": "Segundos", "n_alarmas": "Alarmas", "n_degradados": "Degradados",
                                       "alarmas_en_degradados": "Alarmas en degradados", "n_silenciosos": "Silenciosos", "alarmas_en_silenciosos": "Alarmas en silenciosos"}), ".1f")
    A = rd("m3c_sil_auc.csv")
    A["auc_sep"] = np.maximum(A.auc, 1 - A.auc)
    P = A.pivot(index="variable", columns="grupo", values="auc_sep").reset_index()
    T["sil_auc"] = md(P, ".3f")

    # ---------------------------------------------------------------- 5. control
    SM, TC = rd("m3c_ctl_resumen.csv"), rd("m3c_ctl_tests.csv")
    T["ctl_dec"] = md(SM[["decoder", "ejecuciones", "ensayos", "identicas", "pct_identicas", "ejec_con_alguna_diferencia", "n_samples_iguales_pct",
                          "dif_proba_abs_media"]].rename(columns={"decoder": "Decodificador", "ejecuciones": "Ejecuciones", "ensayos": "Ensayos",
                                                                  "identicas": "Decisiones idénticas", "pct_identicas": "% idénticas",
                                                                  "ejec_con_alguna_diferencia": "Ejecuciones con alguna diferencia",
                                                                  "n_samples_iguales_pct": "% n_samples igual",
                                                                  "dif_proba_abs_media": "|Δ proba| media"}), ".4f")
    T["ctl_tele"] = md(TC.assign(**{"p Wilcoxon": TC.wilcoxon_p.map(pv), "r": TC.r.map(lambda x: "n.d." if pd.isna(x) else f"{x:.2f}"),
                                    "Sujetos mayor/menor en la nueva": TC.apply(lambda r: f"{int(r.n_subj_mayor_nueva)}/{int(r.n_subj_menor_nueva)}", axis=1)})
                       [["variable", "nueva_med_subj", "anterior_med_subj", "dif_med_subj", "dif_ejec_min", "dif_ejec_max", "p Wilcoxon", "r",
                         "Sujetos mayor/menor en la nueva"]]
                       .rename(columns={"variable": "Variable", "nueva_med_subj": "Nueva (mediana entre sujetos)", "anterior_med_subj": "Anterior (mediana entre sujetos)",
                                        "dif_med_subj": "Diferencia (mediana)", "dif_ejec_min": "Dif. por ejecución mín.", "dif_ejec_max": "Dif. por ejecución máx."}), ".5f")
    return T


def main():
    T = build()
    with open(os.path.join(OUT, "m3c_tablas.md"), "w", encoding="utf-8") as f:
        for k, v in T.items():
            f.write(f"<!-- T:{k} -->\n{v}\n\n")
    print({k: len(v) for k, v in T.items()})


if __name__ == "__main__":
    main()
