"""M3b · Tarea C, complemento: caracterización de los segundos 'operativo y degradado' del CASO BASE directamente sobre
analysis/windows.csv (W = 8, umbral = mínimo). EXPLORATORIO.

Salida: analysis-m3b/m3b_C_silenciosa_base.csv (por sujeto y por familia) y consola.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from m3b_common import OUT, ROOT

W = pd.read_csv(os.path.join(ROOT, "analysis", "windows.csv"))
F = W[(W.kind != "none") & W.degraded.notna()]
S = F[(F.operational == 1) & (F.degraded == 1)]
print("segundos evaluables en las 18 condiciones con fallo:", len(F))
print("degradados:", int((F.degraded == 1).sum()), "| operativos y degradados (silenciosos):", len(S),
      f"({len(S) / len(F):.4f} de los evaluables; {len(S) / int((F.degraded == 1).sum()):.3f} de los degradados)")
print("ejecuciones con >= 1 s silencioso:", S.exec_id.nunique(), "de", F.exec_id.nunique())
print(f"de los silenciosos: recepción 100 % {np.mean(S.recv_ratio >= 0.999):.3f} | n_gaps = 0 {np.mean(S.n_gaps == 0):.3f} | "
      f"latencia media < 50 ms {np.mean(S.lat_mean < 0.05):.3f} | las tres {np.mean((S.recv_ratio >= .999) & (S.n_gaps == 0) & (S.lat_mean < .05)):.3f}")
rows = []
for s, g in F.groupby("subject"):
    gs = S[S.subject == s]
    rows.append(dict(grupo="sujeto", clave=int(s), segundos_eval=len(g), segundos_silenciosos=len(gs),
                     ejecuciones_con_silenciosa=gs.exec_id.nunique(), ejecuciones=g.exec_id.nunique()))
for k, g in F.groupby("kind"):
    gs = S[S.kind == k]
    rows.append(dict(grupo="familia", clave=k, segundos_eval=len(g), segundos_silenciosos=len(gs),
                     ejecuciones_con_silenciosa=gs.exec_id.nunique(), ejecuciones=g.exec_id.nunique()))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "m3b_C_silenciosa_base.csv"), index=False)
print(R.to_string(index=False))
