"""M3b · Tarea C: sensibilidad de la divergencia operativo/degradado. EXPLORATORIO, post hoc.

Caso base (Resultados): ventana móvil W = 8 ensayos, umbral por sujeto = mínimo de la balanced accuracy móvil en sus
5 corridas de referencia, operativo = llegan muestras y 0 excepciones en el segundo.
  falla silenciosa = operativo y degradado ; "ruido de alarma" (loud) = no operativo y no degradado.
Se reproduce primero el caso base con las funciones de bcibench.metrics (debe dar silent = 0 en las 18 condiciones y
coincidir con analysis/t_divergencia.csv y analysis/windows.csv) y luego se varía:
  W ∈ {4, 6, 8, 12}
  regla de umbral ∈ {mínimo, percentil 5 de la referencia, mínimo − 0,05}
  operativo ∈ {base, estricto = base y latencia media del segundo < 50 ms y sin huecos (n_gaps = 0)}

Agregación igual que analyze.py: por ejecución → mediana entre corridas por sujeto → mediana entre sujetos (columna
'*_med'). Como la mediana puede ocultar fallas raras, también se informa la proporción AGRUPADA de segundos
(suma de segundos / total, '*_pool') y el número de ejecuciones con silent > 0.

Salidas (analysis-m3b/): m3b_C_base.csv, m3b_C_cond.csv, m3b_C_resumen.csv
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from m3b_common import LABEL_ORDER, OUT, ROOT
from bcibench import metrics as M

WS = [4, 6, 8, 12]
RULES = ["min", "p5", "min-0.05"]
OPS = ["base", "estricto"]


def threshold(refs, rule):
    if rule == "min":
        return M.reference_threshold(refs, 5, "min")
    if rule == "p5":
        return M.reference_threshold(refs, 5, "percentile")
    if rule == "min-0.05":
        return M.reference_threshold(refs, 5, "min") - 0.05
    raise ValueError(rule)


def main():
    execs, trials, tele = M.load_campaign(os.path.join(ROOT, "raw"), "trials.csv")
    assert len(execs) == 855
    execs = execs.sort_values("exec_id").reset_index(drop=True)

    # --------------------------------------------------------- reproducción del caso base
    rolls8 = {e: M.rolling_bacc(trials[e], 8) for e in execs.exec_id}
    th_json = json.load(open(os.path.join(ROOT, "analysis", "thresholds.json")))["thresholds"]
    thr8 = {}
    for s in sorted(execs.subject.unique()):
        refs = execs[(execs.subject == s) & (execs.kind == "none")].exec_id
        thr8[int(s)] = threshold([rolls8[e] for e in refs], "min")
        assert abs(thr8[int(s)] - th_json[str(int(s))]) < 1e-12, (s, thr8[int(s)], th_json[str(int(s))])
    print("umbrales W=8/min reproducen thresholds.json:", {k: round(v, 3) for k, v in thr8.items()})

    Wref = pd.read_csv(os.path.join(ROOT, "analysis", "windows.csv"))
    mine = []
    for _, e in execs.iterrows():
        w = M.window_table(tele[e.exec_id], rolls8[e.exec_id], thr8[int(e.subject)])
        w["exec_id"] = e.exec_id
        mine.append(w[["exec_id", "sec", "operational", "rbacc", "degraded"]])
    mine = pd.concat(mine, ignore_index=True)
    chk = mine.merge(Wref[["exec_id", "sec", "operational", "rbacc", "degraded"]], on=["exec_id", "sec"], suffixes=("", "_ref"))
    assert len(chk) == len(Wref) == len(mine), (len(chk), len(Wref), len(mine))
    same = (np.allclose(chk.rbacc.fillna(-9), chk.rbacc_ref.fillna(-9)) and
            (chk.operational == chk.operational_ref).all() and
            np.array_equal(chk.degraded.fillna(-9), chk.degraded_ref.fillna(-9)))
    print("windows.csv reproducido exactamente (operational, rbacc, degraded):", bool(same), "| filas:", len(chk))
    assert same

    # --------------------------------------------------------- grilla
    cond_rows, exec_rows = [], []
    for W in WS:
        rolls = rolls8 if W == 8 else {e: M.rolling_bacc(trials[e], W) for e in execs.exec_id}
        for rule in RULES:
            thr = {}
            for s in sorted(execs.subject.unique()):
                refs = execs[(execs.subject == s) & (execs.kind == "none")].exec_id
                thr[int(s)] = threshold([rolls[e] for e in refs], rule)
            for _, e in execs.iterrows():
                w = M.window_table(tele[e.exec_id], rolls[e.exec_id], thr[int(e.subject)])
                strict = ((w.operational == 1) & (w.lat_mean.notna()) & (w.lat_mean < 0.050) & (w.n_gaps == 0)).astype(int)
                for op in OPS:
                    ww = w.copy()
                    if op == "estricto":
                        ww["operational"] = strict
                    v = ww.dropna(subset=["degraded"])
                    n = len(v)
                    sil = int(((v.operational == 1) & (v.degraded == 1)).sum())
                    lou = int(((v.operational == 0) & (v.degraded == 0)).sum())
                    exec_rows.append(dict(W=W, rule=rule, op=op, exec_id=e.exec_id, subject=int(e.subject), label=e.label,
                                          kind=e.kind, severity=e.severity, n=n, n_sil=sil, n_lou=lou,
                                          silent=sil / n if n else np.nan, loud=lou / n if n else np.nan,
                                          degraded_rate=float((v.degraded == 1).mean()) if n else np.nan,
                                          operational_rate=float((v.operational == 1).mean()) if n else np.nan))
    X = pd.DataFrame(exec_rows)
    X.to_csv(os.path.join(OUT, "m3b_C_exec.csv"), index=False)

    # --------------------------------------------------------- por condición
    out = []
    for (W, rule, op, lab), g in X.groupby(["W", "rule", "op", "label"]):
        sm = g.groupby("subject")[["silent", "loud", "degraded_rate"]].median()      # mediana entre corridas por sujeto
        out.append(dict(W=W, rule=rule, op=op, label=lab, n_exec=len(g),
                        silent_med=sm.silent.median(), loud_med=sm.loud.median(), degraded_med=sm.degraded_rate.median(),
                        silent_pool=g.n_sil.sum() / g.n.sum(), loud_pool=g.n_lou.sum() / g.n.sum(),
                        n_windows=int(g.n.sum()), n_silent_windows=int(g.n_sil.sum()),
                        n_exec_silent_gt0=int((g.n_sil > 0).sum()),
                        n_subj_silent_med_gt0=int((sm.silent > 0).sum()),
                        silent_max_exec=g.silent.max()))
    C = pd.DataFrame(out)
    C["_o"] = C.label.map({c: i for i, c in enumerate(LABEL_ORDER)})
    C = C.sort_values(["W", "rule", "op", "_o"]).drop(columns="_o")
    C.to_csv(os.path.join(OUT, "m3b_C_cond.csv"), index=False)

    # caso base: comparar con t_divergencia.csv
    base = C[(C.W == 8) & (C.rule == "min") & (C.op == "base")].copy()
    td = pd.read_csv(os.path.join(ROOT, "analysis", "t_divergencia.csv"))
    base = base.merge(td[["label", "silent", "loud"]].rename(columns={"silent": "silent_orig", "loud": "loud_orig"}), on="label")
    base["ok_silent"] = np.isclose(base.silent_med, base.silent_orig, atol=1e-9)
    base["ok_loud"] = np.isclose(base.loud_med, base.loud_orig, atol=1e-9)
    base.to_csv(os.path.join(OUT, "m3b_C_base.csv"), index=False)
    print("\n== CASO BASE (W=8, mínimo, operativo base) vs t_divergencia.csv")
    print(base[["label", "silent_med", "silent_orig", "loud_med", "loud_orig", "silent_pool", "n_silent_windows", "ok_silent", "ok_loud"]]
          .round(4).to_string(index=False))
    assert base.ok_silent.all() and base.ok_loud.all(), "el caso base no reproduce t_divergencia.csv"
    assert (base.silent_med == 0).all()

    # --------------------------------------------------------- resumen por combinación
    res = []
    for (W, rule, op), g in C.groupby(["W", "rule", "op"]):
        f = g[g.label != "ref"]
        r = g[g.label == "ref"].iloc[0]
        res.append(dict(W=W, rule=rule, op=op,
                        n_cond_silent_med_gt0=int((f.silent_med > 0).sum()), max_silent_med=f.silent_med.max(),
                        n_cond_silent_pool_gt0=int((f.silent_pool > 0).sum()), max_silent_pool=f.silent_pool.max(),
                        peor_cond=f.loc[f.silent_pool.idxmax(), "label"] if f.silent_pool.max() > 0 else "",
                        silent_pool_18=f.n_silent_windows.sum() / f.n_windows.sum(),
                        n_exec_silent_gt0_18=int(f.n_exec_silent_gt0.sum()),
                        silent_ref_med=r.silent_med, silent_ref_pool=r.silent_pool, n_exec_silent_ref=int(r.n_exec_silent_gt0),
                        loud_med_max=f.loud_med.max(), loud_pool_18=(f.loud_pool * f.n_windows).sum() / f.n_windows.sum()))
    Rz = pd.DataFrame(res)
    Rz.to_csv(os.path.join(OUT, "m3b_C_resumen.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    print("\n== RESUMEN por combinación (18 condiciones con fallo; 'ref' aparte)")
    print(Rz.round(4).to_string(index=False))
    pos = C[(C.silent_pool > 0) & (C.label != "ref")]
    print("\n== combinaciones × condición con silent > 0 (agrupado)")
    print(pos[["W", "rule", "op", "label", "silent_med", "silent_pool", "n_silent_windows", "n_windows", "n_exec_silent_gt0", "n_subj_silent_med_gt0", "silent_max_exec"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
