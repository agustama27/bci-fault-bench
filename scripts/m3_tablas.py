"""M3: arma las tablas Markdown de la nota Analisis-M3.md a partir de los CSV de analysis-m3/.
Solo formatea (no calcula nada nuevo). Salida: analysis-m3/m3_tablas.md
"""
import os

import numpy as np
import pandas as pd

ROOT = r"C:\Users\agustin.tamagusuku\Desktop\bci-fault-bench\campaign-2026-09-27\analysis-m3"
KIND = {"disconnect": "desconexión uniforme", "disconnect_trial": "desconexión en el ensayo"}


def md(df, fmt=3):
    """Tabla Markdown sin depender de tabulate."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else f"{v:.{fmt}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


out = []
# ---- T1 cortes
R = pd.read_csv(os.path.join(ROOT, "m3_espera_resumen.csv"))
t = pd.DataFrame({
    "Condición": [f"{KIND[k]} {s:g} s" for k, s in zip(R.kind, R.severity)],
    "Cortes": R.n_cortes,
    "d registrada (mediana, s)": R.d_log_med,
    "Hueco en tiempo de muestra: mediana": R.hueco_ts_med,
    "p5": R.hueco_ts_p5, "p95": R.hueco_ts_p95,
    "Espera r = hueco − d: mediana": R.espera_ts_med, "r p5": R.espera_ts_p5, "r p95": R.espera_ts_p95,
    "ia_max − d: mediana": R.espera_ia_med,
})
out.append("### T1\n" + md(t, 3))

# ---- T2 exposición
C = pd.read_csv(os.path.join(ROOT, "m3_exposicion_cond.csv"))
pct = lambda x: (100 * x)
t = pd.DataFrame({
    "Condición": C.label,
    "< 1000, mediana entre sujetos (%)": pct(C.med_subj_lt1000),
    "< 1000, global (%)": pct(C.glob_lt1000),
    "< 500, mediana entre sujetos (%)": pct(C.med_subj_lt500),
    "< 500, global (%)": pct(C.glob_lt500),
    "Inválidos, global (%)": pct(C.glob_invalid),
    "Fracción de ventana perdida (media global, %)": pct(C.lost_frac_glob),
})
out.append("### T2\n" + md(t, 1))

# ---- T3 modelo E
M = C[C.label.str.startswith("disconnect-")]
t = pd.DataFrame({
    "Condición": M.label,
    "d (s)": M.severity,
    "r (tarea 1, s)": M.r_tarea1,
    "Observado: media entre ejecuciones (%)": pct(M.obs_mean_exec),
    "EE (%)": pct(M.obs_se_exec),
    "E con r = 0 (%)": pct(M.E_r0),
    "E = N(d+r+w)/T (%)": pct(M.E_modelo),
    "E con zona de guarda (%)": pct(M.E_guarda_mean),
})
out.append("### T3\n" + md(t, 1))

# ---- T4 exposición vs caída
D = C[C.label != "ref"]
t = pd.DataFrame({
    "Condición": D.label,
    "Ventana tocada (%)": pct(D.glob_lt1000),
    "Fracción perdida (%)": pct(D.lost_frac_glob),
    "ΔBA CSP+LDA (media)": D["dBA_mean_CSP+LDA"],
    "ΔBA CSP+LDA (mediana)": D["dBA_med_CSP+LDA"],
    "p Holm CSP+LDA": D["p_holm_CSP+LDA"],
    "ΔBA EEGNet (media)": D["dBA_mean_EEGNet"],
    "ΔBA EEGNet (mediana)": D["dBA_med_EEGNet"],
    "p Holm EEGNet": D["p_holm_EEGNet"],
})
out.append("### T4\n" + md(t, 3))

# ---- T5 condicional
A = pd.read_csv(os.path.join(ROOT, "m3_condicional.csv"))
A = A[A.label.isin(["disconnect-0.5", "disconnect-1", "disconnect-3"])]
t = pd.DataFrame({
    "Condición": A.label, "Decodificador": A.decoder,
    "Tocados (n)": A.n_t, "Exactitud tocados": A.acc_tocado, "IC95 Wilson tocados": [f"[{a:.2f}; {b:.2f}]" for a, b in zip(A.lo_t, A.hi_t)],
    "No tocados (n)": A.n_u, "Exactitud no tocados": A.acc_no_tocado, "IC95 Wilson no tocados": [f"[{a:.2f}; {b:.2f}]" for a, b in zip(A.lo_u, A.hi_u)],
})
out.append("### T5\n" + md(t, 3))

# ---- T6 IC
I = pd.read_csv(os.path.join(ROOT, "m3_ic.csv"))
for dec in ("CSP+LDA", "EEGNet"):
    g = I[I.decoder == dec]
    t = pd.DataFrame({
        "Condición": g.label,
        "Diferencia media": g.mean_diff,
        "HL": g.HL,
        "IC95 exacto (Wilcoxon)": [f"[{a:+.4f}; {b:+.4f}]".replace("-0.0000", "0.0000").replace("+0.0000", "0.0000") for a, b in zip(g.ci_lo, g.ci_hi)],
        "IC95 bootstrap (HL)": [f"[{a:+.4f}; {b:+.4f}]".replace("-0.0000", "0.0000").replace("+0.0000", "0.0000") for a, b in zip(g.boot_lo, g.boot_hi)],
        "− / 0 / + (sujetos)": [f"{a}/{b}/{c}" for a, b, c in zip(g.n_neg, g.n_zero, g.n_pos)],
        "¿IC exacto dentro de ±0,03?": np.where(g.dentro_exacto, "sí", "NO"),
        "¿IC boot dentro de ±0,03?": np.where(g.dentro_boot, "sí", "NO"),
    })
    out.append(f"### T6 {dec}\n" + md(t, 4))

open(os.path.join(ROOT, "m3_tablas.md"), "w", encoding="utf-8").write("\n\n".join(out))
print("ok")
