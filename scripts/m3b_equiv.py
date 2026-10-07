"""M3b · comparaciones a igual pérdida de muestras con distinta FORMA/UBICACIÓN del hueco (Tarea A, complemento). EXPLORATORIO.

  P1: pérdida uniforme 10 % (≈ 100 muestras sueltas, n ≈ 900) vs. pérdida contigua 10 % (100 muestras seguidas, n = 900).
  P2: pérdida contigua 40 % en posición aleatoria (n ≈ 600) vs. desconexión en el ensayo 0,5 s (hueco de 380 muestras
      pegado al inicio de la ventana, n ≈ 620; 'desc. en ensayo 1 s' da lo mismo por el piso de reconexión).
Por sujeto: tasa de cambio y Δ acierto (acierto − acierto de la referencia en esos ensayos) sobre sus 5 corridas;
Wilcoxon apareado (n = 9). Salida: analysis-m3b/m3b_A_equiv.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from m3b_common import OUT, wilcoxon_paired

D = pd.read_csv(os.path.join(OUT, "m3b_A_trials.csv"))
PAIRS = {"P1 pérdida 10 % uniforme vs contigua": ("loss-random-0.1", "burst_trial-0.1"),
         "P2 contigua 40 % aleatoria vs desc. ensayo 0,5 s (inicio)": ("burst_trial-0.4", "disconnect_trial-0.5")}
rows = []
for name, (a, b) in PAIRS.items():
    for dec, tag in (("CSP+LDA", "csp"), ("EEGNet", "net")):
        s = {}
        for lab in (a, b):
            g = D[D.label_cond == lab]
            s[lab] = g.groupby("subject").agg(chg=(f"chg_{tag}", "mean"), ok=(f"ok_{tag}", "mean"), rok=(f"ref_ok_{tag}", "mean"),
                                              n_med=("n_samples", "median"))
            s[lab]["dacc"] = s[lab].ok - s[lab].rok
        _, _, pc = wilcoxon_paired(s[b].chg.to_numpy(), s[a].chg.to_numpy())
        _, _, pa = wilcoxon_paired(s[b].dacc.to_numpy(), s[a].dacc.to_numpy())
        rows.append(dict(par=name, decoder=dec, A=a, B=b, n_samples_med_A=float(D[D.label_cond == a].n_samples.median()),
                         n_samples_med_B=float(D[D.label_cond == b].n_samples.median()),
                         chg_A=D[D.label_cond == a][f"chg_{tag}"].mean(), chg_B=D[D.label_cond == b][f"chg_{tag}"].mean(),
                         dacc_A=(D[D.label_cond == a][f"ok_{tag}"] - D[D.label_cond == a][f"ref_ok_{tag}"]).mean(),
                         dacc_B=(D[D.label_cond == b][f"ok_{tag}"] - D[D.label_cond == b][f"ref_ok_{tag}"]).mean(),
                         med_subj_diff_chg=float((s[b].chg - s[a].chg).median()), p_chg=pc,
                         subj_B_gt_A=int((s[b].chg > s[a].chg).sum()),
                         med_subj_diff_dacc=float((s[b].dacc - s[a].dacc).median()), p_dacc=pa))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "m3b_A_equiv.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(R.round(4).to_string(index=False))
