"""M3c · Tarea 1: piso de reconexión (campaña exploratoria post hoc, campaign-m3/campana3).

Para cada corte (outage) de las condiciones disconnect-n{2,5,10,20}-1 (d = 1 s) y disconnect-n5-{1.25,1.75,2.25}:
  d_log      = outage_end - outage_start de fault_log.jsonl (grilla de bloques de 40 ms)
  d_plan     = end - start de outage_planned (nominal)
  hueco_ts   = gap_s de la fila de telemetría donde se reanuda el flujo (tiempo de MUESTRA faltante entre la última
               muestra recibida antes del corte y la primera tras él; resolución 40 ms = un bloque de 10 muestras).
               Es la magnitud a la que se ajustó la regla max(1,52; d + 0,52).
  hueco_ia   = ia_max de la misma fila (intervalo entre llegadas, reloj de pared del receptor). Incluye el periodo del
               propio bloque (40 ms) además del hueco: hueco_ia ~ hueco_ts + 0,04 s. Se informa tal cual y corregido (-0,04).
  espera     = hueco - d_log  (datos que el outlet recreado envió y que el receptor no obtuvo)
Regla previa: hueco = max(1,52; d + 0,52). Se evalúa con d_log y con d_plan, tolerancia ±40 ms.

Cada corte se asocia a UNA fila de telemetría con n_gaps > 0 (en orden). Si una ejecución no cumple #filas == #cortes con
n_gaps == 1 en todas las filas, se marca y sus cortes se excluyen de la regla (se informa cuántas).

Salidas (campaign-m3/analysis/): m3c_cortes.csv, m3c_espera_resumen.csv, m3c_regla.csv, m3c_ejecuciones_problema.csv,
m3c_modelo_escalonado.csv, m3c_regla_ajuste_libre.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from m3c_common import FLOOR, NEW, OLD_ROOT, OUT, SWEEP, load_events

TOL = 0.0401
CONDS = [c for c in SWEEP if c != "ref"] + FLOOR


def rule(d, c0=1.52, c1=0.52):
    return np.maximum(c0, np.asarray(d) + c1)


def rule_step(d, c0=1.52, c1=0.52, tick=0.5):
    """Modelo ESCALONADO (post hoc, propuesto a la vista de d = 1,25/1,75/2,25): la reanudación ocurre en el primer
    'tick' de reintento de 0,5 s que sigue a la vuelta del outlet, más 0,52 s fijos, con piso de 1,52 s.
    hueco = max(1,52; 0,52 + 0,5 * ceil(d / 0,5)). Coincide con la regla lineal cuando d es múltiplo de 0,5 s."""
    d = np.asarray(d, float)
    return np.maximum(c0, c1 + tick * np.ceil(np.round(d / tick, 6)))


def main():
    allids = [d for d in os.listdir(NEW) if os.path.isdir(os.path.join(NEW, d))]   # plan.json solo trae las 360 del plan principal
    ids = [e for e in allids if e.split("-", 2)[2] in CONDS]
    rows, problems = [], []
    for eid in sorted(ids):
        d = os.path.join(NEW, eid)
        ev = load_events(os.path.join(d, "fault_log.jsonl"))
        planned = [(e["start"], e["end"]) for e in ev if e["event"] == "outage_planned"]
        starts = [e["t_rel"] for e in ev if e["event"] == "outage_start"]
        ends = [e["t_rel"] for e in ev if e["event"] == "outage_end"]
        tel = pd.read_csv(os.path.join(d, "telemetry.csv"))
        gap_rows = tel[tel.n_gaps > 0].reset_index(drop=True)
        label = eid.split("-", 2)[2]
        subj, run = int(eid[1:3]), eid[5]
        ok = (len(gap_rows) == len(ends) == len(starts) == len(planned)) and bool((gap_rows.n_gaps == 1).all())
        if not ok:
            problems.append(dict(exec_id=eid, planned=len(planned), starts=len(starts), ends=len(ends),
                                 gap_rows=len(gap_rows), sum_n_gaps=int(gap_rows.n_gaps.sum()),
                                 max_n_gaps=int(gap_rows.n_gaps.max()) if len(gap_rows) else 0))
        for k in range(min(len(ends), len(gap_rows))):
            g = gap_rows.iloc[k]
            d_log = ends[k] - starts[k]
            rows.append(dict(exec_id=eid, subject=subj, run=run, label=label, k=k, ok_pairing=ok,
                             start_rel=starts[k], end_rel=ends[k], d_plan=planned[k][1] - planned[k][0], d_log=d_log,
                             n_gaps_row=int(g.n_gaps), sec_resume=int(g.sec),
                             hueco_ts=float(g.gap_s), hueco_ia=float(g.ia_max)))
    C = pd.DataFrame(rows)
    C["d_nom"] = C.d_plan.round(2)
    C["hueco_ia_corr"] = C.hueco_ia - 0.04
    for h in ("hueco_ts", "hueco_ia_corr"):
        C[f"espera_{h}"] = C[h] - C.d_log
    C["pred_dlog"] = rule(C.d_log)
    C["pred_dnom"] = rule(C.d_nom)
    for h in ("hueco_ts", "hueco_ia", "hueco_ia_corr"):
        C[f"err_{h}_dlog"] = C[h] - C.pred_dlog
        C[f"err_{h}_dnom"] = C[h] - C.pred_dnom
    C.to_csv(os.path.join(OUT, "m3c_cortes.csv"), index=False)
    pd.DataFrame(problems).to_csv(os.path.join(OUT, "m3c_ejecuciones_problema.csv"), index=False)
    print(f"ejecuciones analizadas: {len(ids)}  cortes: {len(C)}  ejecuciones con emparejamiento dudoso: {len(problems)} "
          f"({int((~C.ok_pairing).sum())} cortes)")
    if problems:
        print(pd.DataFrame(problems).to_string(index=False))

    # ------------- resumen por condición (solo cortes con emparejamiento fiable)
    Cg = C[C.ok_pairing]
    q = lambda x, p: float(np.percentile(x, p))
    res, reg = [], []
    for lab in CONDS:
        g = Cg[Cg.label == lab]
        res.append(dict(label=lab, n_cortes=len(g), n_cortes_total=int((C.label == lab).sum()),
                        d_plan=g.d_nom.median(), d_log_med=g.d_log.median(),
                        hueco_ts_med=g.hueco_ts.median(), hueco_ts_min=g.hueco_ts.min(), hueco_ts_p5=q(g.hueco_ts, 5),
                        hueco_ts_p95=q(g.hueco_ts, 95), hueco_ts_max=g.hueco_ts.max(),
                        hueco_ia_med=g.hueco_ia.median(), hueco_ia_min=g.hueco_ia.min(), hueco_ia_max=g.hueco_ia.max(),
                        espera_ts_med=g.espera_hueco_ts.median(), espera_ts_min=g.espera_hueco_ts.min(),
                        espera_ts_max=g.espera_hueco_ts.max(),
                        espera_ia_corr_med=g.espera_hueco_ia_corr.median(),
                        pred_dlog_med=g.pred_dlog.median(), pred_dnom_med=g.pred_dnom.median()))
        for h in ("hueco_ts", "hueco_ia", "hueco_ia_corr"):
            for dname in ("dlog", "dnom"):
                e = g[f"err_{h}_{dname}"]
                reg.append(dict(label=lab, hueco=h, d_usado=dname, n=len(g), err_med_ms=1e3 * e.median(),
                                mae_ms=1e3 * e.abs().mean(), max_abs_ms=1e3 * e.abs().max(),
                                pct_dentro_40ms=100 * float((e.abs() <= TOL).mean()),
                                n_dentro=int((e.abs() <= TOL).sum())))
    R, G = pd.DataFrame(res), pd.DataFrame(reg)
    # total (todas las condiciones)
    for h in ("hueco_ts", "hueco_ia", "hueco_ia_corr"):
        for dname in ("dlog", "dnom"):
            e = Cg[f"err_{h}_{dname}"]
            G.loc[len(G)] = dict(label="TODAS", hueco=h, d_usado=dname, n=len(Cg), err_med_ms=1e3 * e.median(),
                                 mae_ms=1e3 * e.abs().mean(), max_abs_ms=1e3 * e.abs().max(),
                                 pct_dentro_40ms=100 * float((e.abs() <= TOL).mean()), n_dentro=int((e.abs() <= TOL).sum()))
    R.to_csv(os.path.join(OUT, "m3c_espera_resumen.csv"), index=False)
    G.to_csv(os.path.join(OUT, "m3c_regla.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(R.round(3).to_string(index=False))
    print(G.round(2).to_string(index=False))

    # ------------- modelo escalonado post hoc: en estos datos y en la campaña anterior (d = 0,5/1/2/3, múltiplos de 0,5)
    OLDC = pd.read_csv(os.path.join(OLD_ROOT, "analysis-m3", "m3_cortes.csv"))
    ST = []
    for nombre, tab, dcol in (("campaña nueva", Cg, "d_nom"), ("campaña anterior", OLDC, "d_plan")):
        for lab, g in tab.groupby("label" if "label" in tab.columns else ["kind", "severity"]):
            for mname, f in (("lineal max(1,52; d+0,52)", rule), ("escalonado max(1,52; 0,52+0,5*ceil(d/0,5))", rule_step)):
                e = g.hueco_ts - f(g[dcol])
                ST.append(dict(campania=nombre, condicion=str(lab), modelo=mname, n=len(g), d_nominal=float(g[dcol].median()),
                               hueco_med=float(g.hueco_ts.median()), pred=float(f(g[dcol]).mean()),
                               mae_ms=1e3 * float(e.abs().mean()), max_abs_ms=1e3 * float(e.abs().max()),
                               pct_dentro_40ms=100 * float((e.abs() <= TOL).mean())))
        for mname, f in (("lineal max(1,52; d+0,52)", rule), ("escalonado max(1,52; 0,52+0,5*ceil(d/0,5))", rule_step)):
            e = tab.hueco_ts - f(tab[dcol])
            ST.append(dict(campania=nombre, condicion="TODAS", modelo=mname, n=len(tab), d_nominal=np.nan,
                           hueco_med=float(tab.hueco_ts.median()), pred=np.nan, mae_ms=1e3 * float(e.abs().mean()),
                           max_abs_ms=1e3 * float(e.abs().max()), pct_dentro_40ms=100 * float((e.abs() <= TOL).mean())))
    ST = pd.DataFrame(ST)
    ST.to_csv(os.path.join(OUT, "m3c_modelo_escalonado.csv"), index=False)
    print("\nlineal vs escalonado (hueco_ts, d nominal)")
    print(ST.round(2).to_string(index=False))

    # ------------- valores discretos
    print("\nvalores de hueco_ts por condición (valor: n)")
    for lab in CONDS:
        g = Cg[Cg.label == lab]
        print(lab, {k: int(v) for k, v in g.hueco_ts.round(2).value_counts().sort_index().items()}, "| d_log:", {k: int(v) for k, v in g.d_log.round(2).value_counts().sort_index().items()})

    # ------------- ajuste libre max(c0, d + c1) con d_log, para contrastar con la regla fija
    best = None
    for c0 in np.arange(1.0, 2.0, 0.01):
        for c1 in np.arange(0.0, 1.0, 0.01):
            sse = float(((Cg.hueco_ts - np.maximum(c0, Cg.d_log + c1)) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, c0, c1)
    _, c0, c1 = best
    e = Cg.hueco_ts - np.maximum(c0, Cg.d_log + c1)
    print(f"\najuste libre hueco_ts = max(c0, d_log + c1): c0={c0:.2f} c1={c1:.2f}  MAE={1e3*e.abs().mean():.1f} ms max|err|={1e3*e.abs().max():.0f} ms")
    pd.DataFrame([dict(c0=c0, c1=c1, mae_ms=1e3 * e.abs().mean(), max_abs_ms=1e3 * e.abs().max(), n=len(Cg))]).to_csv(
        os.path.join(OUT, "m3c_regla_ajuste_libre.csv"), index=False)


if __name__ == "__main__":
    main()
