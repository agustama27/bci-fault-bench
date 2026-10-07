"""M3b · control de la Tarea A: ¿los ensayos 'no tocados' (ventana de 4 s íntegra) tienen huecos en el RELLENO de 1 s
previo/posterior que usa el filtro IIR? Si los hubiera y las decisiones siguieran idénticas, el cero de 'no tocados' no
depende de la definición estricta de ventana. EXPLORATORIO.

Resolución: telemetry.csv es por segundo; la ventana del ensayo es [onset+2, onset+6) y el relleno [onset+1, onset+2) y
[onset+6, onset+7). Con la ventana íntegra (n_samples = 1000), un déficit de muestras en el segundo floor(onset+1) o
floor(onset+6) solo puede venir del relleno (salvo 3 muestras de borde: se exigen >= 4 faltantes).

Salida: analysis-m3b/m3b_A_pad.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from m3b_common import OUT, ROOT

D = pd.read_csv(os.path.join(OUT, "m3b_A_trials.csv"))
U = D[(D.kind == "disconnect") & (~D.touched)]
rows = []
for eid, g in U.groupby("exec_id"):
    te = pd.read_csv(os.path.join(ROOT, "raw", eid, "telemetry.csv")).set_index("sec")
    for _, r in g.iterrows():
        pre, post = int(np.floor(r.onset + 1)), int(np.floor(r.onset + 6))
        d_pre = 250 - int(te.n_samples.get(pre, 250)); d_post = 250 - int(te.n_samples.get(post, 250))
        rows.append(dict(exec_id=eid, i=int(r.i), subject=int(r.subject), severity=r.severity, d_pre=d_pre, d_post=d_post,
                         pad_tocado=bool(d_pre >= 4 or d_post >= 4), chg_csp=int(r.chg_csp), chg_net=int(r.chg_net)))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(OUT, "m3b_A_pad.csv"), index=False)
print("ensayos no tocados en desconexión uniforme:", len(P))
print(P.groupby("pad_tocado").agg(n=("i", "size"), chg_csp=("chg_csp", "sum"), chg_net=("chg_net", "sum")))
print("por severidad (pad tocado):")
print(P[P.pad_tocado].groupby("severity").agg(n=("i", "size"), chg_csp=("chg_csp", "sum"), chg_net=("chg_net", "sum")))
