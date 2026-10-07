"""M3 · Tarea 1: piso de reconexión del inlet LSL tras un corte (outlet recreado).

Para cada corte individual de las ejecuciones disconnect y disconnect_trial:
  hueco_ts  = gap_s de la fila de telemetría donde se reanuda el flujo
              (tiempo de muestra entre la última muestra recibida antes del corte y la
               primera recibida después, en la escala nominal; resolución 40 ms = un bloque de
               10 muestras). gap_s ya viene calculado por consumer.py: sum(dt - 1/fs) sobre
               los saltos de marca de tiempo > 1.5/fs.
  hueco_ia  = ia_max de la misma fila (intervalo entre llegadas, reloj de pared del receptor).
  d         = duración del corte registrada (outage_end - outage_start de fault_log.jsonl,
              alineada a la grilla de bloques de 40 ms).
  espera_receptor = hueco_ts - d  (datos enviados tras recrear el outlet que el receptor no recibió).

Sale en campaign-2026-09-27/analysis-m3/:
  m3_cortes.csv            una fila por corte
  m3_espera_resumen.csv    mediana, p5, p95 por (kind, severidad)
  m3_grilla.csv            pruebas de grilla / modelo max(c0, d + c1)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\agustin.tamagusuku\Desktop\bci-fault-bench\campaign-2026-09-27"
OUT = os.path.join(ROOT, "analysis-m3")
os.makedirs(OUT, exist_ok=True)
SFREQ = 250.0
CHUNK_S = 10 / SFREQ


def load_events(path):
    ev = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    planned = [(e["start"], e["end"]) for e in ev if e["event"] == "outage_planned"]
    starts = [e["t_rel"] for e in ev if e["event"] == "outage_start"]
    ends = [e["t_rel"] for e in ev if e["event"] == "outage_end"]
    return planned, starts, ends


def main():
    execs = pd.read_csv(os.path.join(ROOT, "analysis", "execs.csv"))
    sel = execs[execs.kind.isin(["disconnect", "disconnect_trial"])]
    rows, problems = [], []
    for _, e in sel.iterrows():
        d = os.path.join(ROOT, "raw", e.exec_id)
        planned, starts, ends = load_events(os.path.join(d, "fault_log.jsonl"))
        tel = pd.read_csv(os.path.join(d, "telemetry.csv"))
        prod = json.load(open(os.path.join(d, "producer.json")))
        # el último corte puede no tener outage_end si la corrida termina dentro
        n_end = len(ends)
        gap_rows = tel[tel.n_gaps > 0].reset_index(drop=True)
        if len(gap_rows) != n_end or (gap_rows.n_gaps != 1).any():
            problems.append((e.exec_id, len(planned), len(starts), n_end, len(gap_rows), int(gap_rows.n_gaps.sum())))
        for k in range(min(n_end, len(gap_rows))):
            g = gap_rows.iloc[k]
            d_log = ends[k] - starts[k]
            # fila siguiente al hueco: la llegada previa al corte está en una fila anterior;
            # el ia_max de la fila de reanudación es el intervalo entre llegadas que cruza el corte
            rows.append(dict(
                exec_id=e.exec_id, subject=e.subject, run=e.run, kind=e.kind, severity=e.severity,
                k=k, start_rel=starts[k], end_rel=ends[k], d_plan=planned[k][1] - planned[k][0],
                d_log=d_log, sec_resume=int(g.sec),
                hueco_ts=float(g.gap_s), hueco_ia=float(g.ia_max),
                espera_ts=float(g.gap_s) - d_log, espera_ia=float(g.ia_max) - d_log,
                t_resume_rel=starts[k] + float(g.gap_s),     # instante nominal de la 1.a muestra recibida tras el corte
            ))
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(OUT, "m3_cortes.csv"), index=False)
    print(f"cortes: {len(C)}  ejecuciones: {sel.shape[0]}  con discrepancia: {len(problems)}")
    for p in problems[:20]:
        print("  (exec, planned, starts, ends, gap_rows, sum n_gaps) =", p)

    # ---- resumen por tipo y severidad
    def q(x, p):
        return float(np.percentile(x, p))
    res = []
    for (kind, sev), g in C.groupby(["kind", "severity"]):
        res.append(dict(kind=kind, severity=sev, n_cortes=len(g),
                        d_log_med=g.d_log.median(),
                        hueco_ts_med=g.hueco_ts.median(), hueco_ts_p5=q(g.hueco_ts, 5), hueco_ts_p95=q(g.hueco_ts, 95),
                        hueco_ia_med=g.hueco_ia.median(),
                        espera_ts_med=g.espera_ts.median(), espera_ts_p5=q(g.espera_ts, 5), espera_ts_p95=q(g.espera_ts, 95),
                        espera_ts_min=g.espera_ts.min(), espera_ts_max=g.espera_ts.max(),
                        espera_ia_med=g.espera_ia.median(), espera_ia_p5=q(g.espera_ia, 5), espera_ia_p95=q(g.espera_ia, 95),
                        hueco_ts_min=g.hueco_ts.min(), hueco_ts_max=g.hueco_ts.max()))
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "m3_espera_resumen.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(R.round(3).to_string())

    # ---- valores discretos del hueco_ts
    print("\nvalores distintos de hueco_ts por severidad (valor: n)")
    for (kind, sev), g in C.groupby(["kind", "severity"]):
        vc = g.hueco_ts.round(2).value_counts().sort_index()
        print(kind, sev, dict(vc.head(15)))

    # ---- modelo hueco = max(c0, d + c1): ajuste por mínimos cuadrados con grid
    best = None
    for c0 in np.arange(1.0, 2.0, 0.01):
        for c1 in np.arange(0.0, 1.0, 0.01):
            pred = np.maximum(c0, C.d_log + c1)
            sse = float(((C.hueco_ts - pred) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, c0, c1)
    sse, c0, c1 = best
    pred = np.maximum(c0, C.d_log + c1)
    mae = float(np.abs(C.hueco_ts - pred).mean())
    print(f"\najuste hueco = max(c0, d+c1): c0={c0:.2f} c1={c1:.2f}  MAE={mae*1000:.1f} ms  max|err|={np.abs(C.hueco_ts-pred).max()*1000:.0f} ms")
    # modelos fijos del enunciado
    for name, pr in [("max(1,56, d+0,56) [ia]", np.maximum(1.56, C.d_log + 0.56)),
                     ("max(1,52, d+0,52) [ts]", np.maximum(1.52, C.d_log + 0.52))]:
        tgt = C.hueco_ia if "ia" in name else C.hueco_ts
        err = tgt - pr
        print(f"{name}: MAE={np.abs(err).mean()*1000:.1f} ms, max|err|={np.abs(err).max()*1000:.0f} ms, "
              f"|err|<=40 ms en {100*(np.abs(err)<=0.0401).mean():.1f}% de los cortes")

    # ---- grilla: ¿el instante de reanudación (start+hueco) o la espera se concentra en valores discretos?
    G = []
    for (kind, sev), g in C.groupby(["kind", "severity"]):
        for name, v in [("espera_ts", g.espera_ts), ("hueco_ts", g.hueco_ts),
                        ("t_resume_rel mod 0.5", g.t_resume_rel % 0.5), ("t_resume_rel mod 1", g.t_resume_rel % 1.0),
                        ("start_rel mod 0.5", g.start_rel % 0.5), ("end_rel mod 0.5", g.end_rel % 0.5)]:
            vr = np.round(v, 2)
            nuniq = vr.nunique()
            top = vr.value_counts(normalize=True).head(3)
            G.append(dict(kind=kind, severity=sev, variable=name, n=len(v), valores_distintos=nuniq,
                          moda=float(top.index[0]), frec_moda=float(top.iloc[0]),
                          sd_ms=float(v.std() * 1000), rango_ms=float((v.max() - v.min()) * 1000)))
    G = pd.DataFrame(G)
    G.to_csv(os.path.join(OUT, "m3_grilla.csv"), index=False)
    print("\n", G.round(3).to_string())

    # ---- por sujeto/corrida: ¿depende de algo? (espera_ts vs d)
    print("\nespera_ts por (kind, severity, subject) media:")
    print(C.pivot_table(index="subject", columns=["kind", "severity"], values="espera_ts", aggfunc="median").round(2).to_string())
    print("\nSpearman espera_ts vs sec_resume, start_rel dentro de cada condición:")
    from scipy import stats
    for (kind, sev), g in C.groupby(["kind", "severity"]):
        r1 = stats.spearmanr(g.espera_ts, g.start_rel)
        print(kind, sev, f"rho(espera,start_rel)={r1.statistic:.3f} p={r1.pvalue:.3f}",
              f"| rho(espera, d_plan)={'n/a'}")


if __name__ == "__main__":
    main()
