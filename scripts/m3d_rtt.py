"""M3d · Confirmación del mecanismo del piso de reconexión (campaña exploratoria post hoc).

Mismas condiciones (disconnect-n5 con d = 0,5/1,1/1,25/1,75 s, sujetos 1-9, corrida 0) bajo dos valores de
tuning.MulticastMinRTT de liblsl (lsl_api.cfg vía LSLAPICFG): 0,5 s (valor por defecto) y 0,25 s.

Mecanismo (liblsl v1.17.7, idéntico en master):
  - inlet_connection.cpp: resolve_oneshot(query, 1, FOREVER, attempt == 0 ? 1.0 : 5.0)  -> espera mínima de 1 s
  - resolver_impl.cpp: en resolve_oneshot (fast_mode) las consultas salen cada multicast_min_rtt
  - data_receiver.cpp: sleep_for(500 ms) después de recuperar, antes de reconectar
Predicción: hueco = max(1,52; 0,52 + RTT * ceil(d / RTT)).

hueco_ts = gap_s de la fila de telemetría donde se reanuda el flujo (misma magnitud que m3c_reconexion.py).
Uso: python scripts/m3d_rtt.py --roots campaign-m3/rtt050 campaign-m3/rtt025 --rtt 0.5 0.25 --out campaign-m3/analysis
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

TOL = 0.0401


def pred(d, rtt, c0=1.52, c1=0.52):
    d = np.asarray(d, float)
    return np.maximum(c0, c1 + rtt * np.ceil(np.round(d / rtt, 6)))


def events(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+", required=True)
ap.add_argument("--rtt", nargs="+", type=float, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rows, bad = [], []
for root, rtt in zip(a.roots, a.rtt):
    for eid in sorted(os.listdir(root)):
        d = os.path.join(root, eid)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "telemetry.csv")):
            continue
        ev = events(os.path.join(d, "fault_log.jsonl"))
        planned = [(e["start"], e["end"]) for e in ev if e["event"] == "outage_planned"]
        starts = [e["t_rel"] for e in ev if e["event"] == "outage_start"]
        ends = [e["t_rel"] for e in ev if e["event"] == "outage_end"]
        tel = pd.read_csv(os.path.join(d, "telemetry.csv"))
        g = tel[tel.n_gaps > 0].reset_index(drop=True)
        ok = len(g) == len(ends) == len(starts) == len(planned) and bool((g.n_gaps == 1).all())
        if not ok:
            bad.append(dict(rtt=rtt, exec_id=eid, planned=len(planned), ends=len(ends), gap_rows=len(g)))
            continue
        for k in range(len(ends)):
            rows.append(dict(rtt=rtt, exec_id=eid, d_nom=round(planned[k][1] - planned[k][0], 2),
                             d_log=ends[k] - starts[k], hueco_ts=float(g.gap_s[k])))
C = pd.DataFrame(rows)
C["pred_rtt050"] = pred(C.d_nom, 0.5)
C["pred_rtt025"] = pred(C.d_nom, 0.25)
C["pred_propia"] = [pred(d, r) for d, r in zip(C.d_nom, C.rtt)]
C["err_ms"] = 1e3 * (C.hueco_ts - C.pred_propia)
# Con d registrada (grilla de bloques de 40 ms): resuelve la fase cuando d cae sobre un punto de la grilla
C["pred_dlog"] = [pred(d, r) for d, r in zip(C.d_log, C.rtt)]
C["ok_dlog"] = (C.hueco_ts - C.pred_dlog).abs() <= TOL
C.to_csv(os.path.join(a.out, "m3d_rtt_cortes.csv"), index=False)

def resumen(x):
    return pd.Series(dict(
        n=len(x), hueco_med=x.hueco_ts.median(), hueco_min=x.hueco_ts.min(), hueco_max=x.hueco_ts.max(),
        pred_rtt050=x.pred_rtt050.iloc[0], pred_rtt025=x.pred_rtt025.iloc[0],
        dentro_pred_propia=100 * float((x.err_ms.abs() <= 1e3 * TOL).mean()),
        dentro_pred_050=100 * float(((x.hueco_ts - x.pred_rtt050).abs() <= TOL).mean()),
        dentro_pred_025=100 * float(((x.hueco_ts - x.pred_rtt025).abs() <= TOL).mean())))


R = C.groupby(["rtt", "d_nom"]).apply(resumen).reset_index()
R.to_csv(os.path.join(a.out, "m3d_rtt_resumen.csv"), index=False)
pd.set_option("display.width", 200)
print(f"cortes: {len(C)}  ejecuciones descartadas por emparejamiento: {len(bad)}")
if bad:
    print(pd.DataFrame(bad).to_string(index=False))
print(R.round(3).to_string(index=False))
print(f"\nmodelo con d registrada: {int(C.ok_dlog.sum())} de {len(C)} cortes dentro de ±40 ms")
print(pd.crosstab([C.rtt, C.d_nom, C.d_log.round(2)], C.hueco_ts.round(2)))
print("\nvalores de hueco_ts (valor: n)")
for (r, d), x in C.groupby(["rtt", "d_nom"]):
    print(r, d, {k: int(v) for k, v in x.hueco_ts.round(2).value_counts().sort_index().items()})
