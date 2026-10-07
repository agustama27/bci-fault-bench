"""M3b · genera los fragmentos Markdown de la nota Analisis-M3-ensayos-divergencia.md a partir de los CSV de
analysis-m3b/ (evita transcribir cifras a mano). Salida: analysis-m3b/m3b_tablas.md. Solo lectura de CSV."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from m3b_common import LABEL_ORDER, OUT

ES = {"ref": "referencia", "loss-random-0.01": "pérdida 1 %", "loss-random-0.05": "pérdida 5 %", "loss-random-0.1": "pérdida 10 %",
      "jitter-0.01": "jitter 10 ms", "jitter-0.05": "jitter 50 ms", "jitter-0.1": "jitter 100 ms", "delay-0.05": "retraso 50 ms",
      "delay-0.1": "retraso 100 ms", "delay-0.25": "retraso 250 ms", "disconnect-0.5": "desconexión 0,5 s",
      "disconnect-1": "desconexión 1 s", "disconnect-3": "desconexión 3 s", "burst_trial-0.1": "pérdida contigua 10 %",
      "burst_trial-0.25": "pérdida contigua 25 %", "burst_trial-0.4": "pérdida contigua 40 %",
      "disconnect_trial-0.5": "desc. en ensayo 0,5 s", "disconnect_trial-1": "desc. en ensayo 1 s",
      "disconnect_trial-2": "desc. en ensayo 2 s"}


def pct(x, d=1):
    return "—" if pd.isna(x) else f"{100 * x:.{d}f}"


def pp(x, d=1):
    return "—" if pd.isna(x) else f"{100 * x:+.{d}f}"


def pv(x):
    if pd.isna(x):
        return "—"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def tab(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s


def main():
    out = []
    # ---------------------------------------------------------------- A1 / A2
    C = pd.read_csv(os.path.join(OUT, "m3b_A_cond.csv"))
    for dec in ("CSP+LDA", "EEGNet"):
        c = C[C.decoder == dec].set_index("cond")
        rows = []
        for lab in LABEL_ORDER[1:]:
            r = c.loc[lab]
            rows.append([ES[lab], int(r.n_tocado), int(r.n_no_tocado), pct(r.chg_tocado), pct(r.chg_no_tocado),
                         pct(r.acc_tocado), pct(r.acc_ref_mismos_tocado), pp(r.d_acc_tocado),
                         pct(r.harm_tocado), pct(r.help_tocado), pp(r.med_subj_dacc_tocado), pv(r.p_dacc_vs_ref_tocado)])
        out.append(f"### TABLA A1 · {dec}\n\n" + tab(
            ["Condición", "Ens. tocados", "Ens. no tocados", "Cambio tocados (%)", "Cambio no tocados (%)", "Acierto tocados (%)",
             "Acierto de la ref. en esos ensayos (%)", "Δ acierto (pp)", "Perjudicial (%)", "Favorable (%)",
             "Δ acierto mediana sujeto (pp)", "p (Wilcoxon sujeto, Δ vs ref)"], rows))
        rows = []
        for lab in ("disconnect-0.5", "disconnect-1", "disconnect-3", "disconnect-* (3)"):
            r = c.loc[lab]
            rows.append([ES.get(lab, "desconexión uniforme (3 sev. juntas)"), int(r.n_subj_ambos), pct(r.med_subj_chg_tocado),
                         pct(r.med_subj_chg_no_tocado), pv(r.p_chg_t_vs_u), f"{int(r.subj_chg_t_gt_u)}/{int(r.n_subj_ambos)}",
                         pct(r.med_subj_diff_acc), pv(r.p_acc_t_vs_u), pp(r.med_subj_dacc_tocado), pv(r.p_dacc_vs_ref_tocado)])
        out.append(f"### TABLA A2 · {dec}: contraste tocado vs. no tocado (nivel sujeto)\n\n" + tab(
            ["Condición", "Sujetos con ambos tipos", "Cambio tocados (mediana suj., %)", "Cambio no tocados (mediana suj., %)",
             "p cambio (Wilcoxon apareado)", "Sujetos con más cambio en tocados", "Δ acierto toc. − no toc. (mediana suj., pp)",
             "p acierto (Wilcoxon apareado)", "Δ acierto toc. vs ref. (mediana suj., pp)", "p (vs ref.)"], rows))

    # ---------------------------------------------------------------- A3
    D = pd.read_csv(os.path.join(OUT, "m3b_A_dosis.csv"))
    rows = []
    for dec in ("CSP+LDA", "EEGNet"):
        for _, r in D[D.decoder == dec].iterrows():
            rows.append([dec, r.bin, int(r.n_trials), int(r.n_subj), f"{pct(r.chg)} [{pct(r.chg_lo)}; {pct(r.chg_hi)}]",
                         pct(r.med_subj_chg) + (f" ({pct(r.q1_subj_chg)}–{pct(r.q3_subj_chg)})" if not pd.isna(r.q1_subj_chg) else ""),
                         pct(r.harm), pct(r.help), pct(r.acc), pct(r.acc_ref_mismos), pp(r.acc - r.acc_ref_mismos),
                         pv(r.p_chg_vs1000_holm), pv(r.p_acc_vs1000_holm)])
    out.append("### TABLA A3 · dosis-respuesta (18 condiciones con fallo juntas)\n\n" + tab(
        ["Decodificador", "n_samples", "Ensayos", "Sujetos", "Cambio agrupado % [IC Wilson 95 %]", "Cambio mediana suj. % (Q1–Q3)",
         "Perjudicial %", "Favorable %", "Acierto %", "Acierto ref. mismos ens. %", "Δ (pp)", "p Holm cambio vs bin 1000",
         "p Holm acierto vs bin 1000"], rows))
    Fz = pd.read_csv(os.path.join(OUT, "m3b_A_dosis_familia.csv"))
    piv = Fz.pivot_table(index=["family", "bin"], columns="decoder", values=["n_trials", "chg", "acc", "acc_ref_mismos"]).reset_index()
    order_bin = {b: i for i, b in enumerate(["1000", "900-999", "750-899", "500-749", "125-499", "inválido"])}
    piv["_o"] = piv[("bin", "")].map(order_bin)
    piv = piv.sort_values([("family", ""), "_o"])
    rows = []
    for _, r in piv.iterrows():
        rows.append([r[("family", "")], r[("bin", "")], int(r[("n_trials", "CSP+LDA")]),
                     pct(r[("chg", "CSP+LDA")]), pp(r[("acc", "CSP+LDA")] - r[("acc_ref_mismos", "CSP+LDA")]),
                     pct(r[("chg", "EEGNet")]), pp(r[("acc", "EEGNet")] - r[("acc_ref_mismos", "EEGNet")])])
    out.append("### TABLA A4 · dosis-respuesta por familia\n\n" + tab(
        ["Familia", "n_samples", "Ensayos", "Cambio CSP+LDA %", "Δ acierto CSP+LDA (pp)", "Cambio EEGNet %", "Δ acierto EEGNet (pp)"], rows))

    # ---------------------------------------------------------------- B
    V = pd.read_csv(os.path.join(OUT, "m3b_B_validacion.csv"))
    g = V.groupby("severity").agg(execs=("exec_id", "size"), trials=("n_trials", "sum"), matched=("n_match", "sum"),
                                  all_ok=("pct", lambda s: int((s == 1).sum()))).reset_index()
    out.append("### TABLA B0 · validación de la posición recuperada contra la telemetría\n\n" + tab(
        ["Severidad", "Ejecuciones", "Ensayos", "Ensayos con déficit por segundo idéntico", "Ejecuciones con 24/24"],
        [[f"{r.severity:g}", r.execs, r.trials, r.matched, r.all_ok] for r in g.itertuples()]))
    R = pd.read_csv(os.path.join(OUT, "m3b_B_posicion.csv"))
    R = R[R.clasif == "pos"]
    rows = []
    for sev in ["0.1", "0.25", "0.4", "todas"]:
        for pos in ["inicio", "medio", "fin"]:
            a = R[(R.severity.astype(str) == sev) & (R.pos == pos)]
            cs, en = a[a.decoder == "CSP+LDA"].iloc[0], a[a.decoder == "EEGNet"].iloc[0]
            rows.append([sev if sev != "todas" else "todas", pos, int(cs.n), pct(cs.chg), pp(cs.d_acc), pct(en.chg), pp(en.d_acc),
                         pct(en.harm), pct(en.help)])
    out.append("### TABLA B1 · posición del hueco × severidad (burst_trial; tercios de u_frac)\n\n" + tab(
        ["Severidad", "Posición", "Ensayos", "Cambio CSP+LDA %", "Δ acierto CSP+LDA (pp)", "Cambio EEGNet %", "Δ acierto EEGNet (pp)",
         "EEGNet perjudicial %", "EEGNet favorable %"], rows))
    P = pd.read_csv(os.path.join(OUT, "m3b_B_pruebas.csv"))
    P = P[P.metric.isin(["chg", "acc", "dacc"])]
    rows = []
    for dec in ("CSP+LDA", "EEGNet"):
        for sev in ["0.1", "0.25", "0.4", "todas (prom.)"]:
            for m in ("chg", "dacc"):
                r = P[(P.decoder == dec) & (P.severity.astype(str) == sev) & (P.metric == m)].iloc[0]
                rows.append([dec, sev, {"chg": "cambio", "dacc": "Δ acierto"}[m], int(r.n_subj), pct(r.med_inicio), pct(r.med_medio),
                             pct(r.med_fin), pv(r.friedman_p), pv(r.p_inicio_vs_fin)])
    out.append("### TABLA B2 · pruebas por sujeto (n = 9): Friedman entre posiciones y Wilcoxon inicio vs. fin\n\n" + tab(
        ["Decodificador", "Severidad", "Variable", "Sujetos", "Mediana inicio %", "Mediana medio %", "Mediana fin %",
         "p Friedman", "p Wilcoxon inicio vs fin"], rows))

    # ---------------------------------------------------------------- C
    X = pd.read_csv(os.path.join(OUT, "m3b_C_cond.csv"))
    b = X[(X.W == 8) & (X.rule == "min") & (X.op == "base")].set_index("label")
    rows = [[ES[l], pct(b.loc[l].silent_med, 2), pct(b.loc[l].silent_pool, 2), int(b.loc[l].n_silent_windows), int(b.loc[l].n_windows),
             f"{int(b.loc[l].n_exec_silent_gt0)}/{int(b.loc[l].n_exec)}", int(b.loc[l].n_subj_silent_med_gt0), pct(b.loc[l].silent_max_exec),
             pct(b.loc[l].loud_med, 2)] for l in LABEL_ORDER]
    out.append("### TABLA C1 · caso base (W = 8, umbral = mínimo, operativo base)\n\n" + tab(
        ["Condición", "Silenciosa: mediana entre sujetos (%)", "Silenciosa: segundos agrupados (%)", "Segundos silenciosos",
         "Segundos evaluables", "Ejecuciones con ≥ 1 s silencioso", "Sujetos con mediana > 0", "Máximo en una ejecución (%)",
         "'No operativo y no degradado': mediana (%)"], rows))
    Rz = pd.read_csv(os.path.join(OUT, "m3b_C_resumen.csv"))
    nm = {"min": "mínimo", "p5": "percentil 5", "min-0.05": "mínimo − 0,05"}
    rows = []
    for r in Rz.sort_values(["rule", "op", "W"], key=lambda s: s.map({"min": 0, "min-0.05": 1, "p5": 2, "base": 0, "estricto": 1}) if s.name in ("rule", "op") else s).itertuples():
        rows.append([nm[r.rule], r.op, r.W, r.n_cond_silent_med_gt0, pct(r.max_silent_med, 1), r.n_cond_silent_pool_gt0, pct(r.max_silent_pool, 1),
                     pct(r.silent_pool_18, 2), r.n_exec_silent_gt0_18, pct(r.silent_ref_pool, 2), r.n_exec_silent_ref, pct(r.loud_pool_18, 1)])
    out.append("### TABLA C2 · resumen de la grilla (18 condiciones con fallo; 'ref.' = referencia)\n\n" + tab(
        ["Umbral", "Operativo", "W", "Cond. con mediana > 0 (de 18)", "Máx. mediana (%)", "Cond. con segundos agrupados > 0 (de 18)",
         "Máx. agrupado (%)", "Agrupado en las 18 (%)", "Ejecuciones con silenciosa (de 810)", "Ref.: agrupado (%)", "Ref.: ejecuciones (de 45)",
         "'No operativo y no degradado' agrupado, 18 cond. (%)"], rows))
    for rule in ("min", "min-0.05", "p5"):
        p = X[(X.rule == rule) & (X.op == "base")].pivot(index="label", columns="W", values="silent_pool").loc[LABEL_ORDER]
        rows = [[ES[l]] + [pct(p.loc[l, w], 2) for w in (4, 6, 8, 12)] for l in LABEL_ORDER]
        out.append(f"### TABLA C3 · segundos silenciosos agrupados (%) por W, umbral {nm[rule]}, operativo base\n\n" + tab(
            ["Condición", "W = 4", "W = 6", "W = 8", "W = 12"], rows))
    p = X[(X.rule == "min") & (X.op == "estricto") & (X.W == 8)].set_index("label").loc[LABEL_ORDER]
    rows = [[ES[l], pct(p.loc[l].silent_pool, 2), pct(p.loc[l].loud_pool, 1), pct(b.loc[l].loud_pool, 1)] for l in LABEL_ORDER]
    out.append("### TABLA C4 · operativo estricto (latencia < 50 ms y sin huecos), W = 8, mínimo\n\n" + tab(
        ["Condición", "Silenciosa agrupada (%)", "'No operativo y no degradado' estricto (%)", "'No operativo y no degradado' base (%)"], rows))
    with open(os.path.join(OUT, "m3b_tablas.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("ok", os.path.join(OUT, "m3b_tablas.md"))


if __name__ == "__main__":
    main()
