"""Compara dos ejecuciones de la misma corrida y condición con consumidores distintos
(propio vs. BciPy): decisiones por ensayo, segmentos recibidos y telemetría.

Uso: python scripts/compare_consumers.py <carpeta_own> <carpeta_bcipy>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("a", help="ejecución de referencia (consumidor propio)")
ap.add_argument("b", help="ejecución a comparar (consumidor BciPy)")
ap.add_argument("--onset-tol", type=float, default=1e-3, help="tolerancia para emparejar ensayos (s)")
a = ap.parse_args()


def consumer_name(d):
    p = os.path.join(d, "consumer.json")
    return json.load(open(p)).get("consumer", "own") if os.path.exists(p) else "?"


def tel_summary(d):
    t = pd.read_csv(os.path.join(d, "telemetry.csv"))
    return {"recv_ratio": t.n_samples.sum() / t.n_expected.sum(), "n_gaps": int(t.n_gaps.sum()),
            "gap_s": float(t.gap_s.sum()), "lat_mean": float(t.lat_mean.mean()),
            "lat_max": float(t.lat_max.max()), "chunks_per_s": float(t.n_chunks.mean()),
            "exceptions": int(t.exceptions.sum()), "seconds": len(t)}


ta, tb = pd.read_csv(os.path.join(a.a, "trials.csv")), pd.read_csv(os.path.join(a.b, "trials.csv"))
na, nb = consumer_name(a.a), consumer_name(a.b)
m = pd.merge_asof(ta.sort_values("onset"), tb.sort_values("onset"), on="onset", direction="nearest",
                  tolerance=a.onset_tol, suffixes=("_a", "_b")).dropna(subset=["label_b"])
both = m[(m.valid_a == 1) & (m.valid_b == 1)]
agree = (both.pred_a.astype(float) == both.pred_b.astype(float))
dd = (m.decision_delay_b - m.decision_delay_a)

print(f"A = {a.a}  [{na}]\nB = {a.b}  [{nb}]\n")
print("Ensayos")
rows = [
    ("n_trials", len(ta), len(tb), ""),
    ("emparejados por onset", len(m), len(m), f"tol {a.onset_tol} s"),
    ("etiqueta coincide", int((m.label_a == m.label_b).sum()), int((m.label_a == m.label_b).sum()), f"de {len(m)}"),
    ("válidos", int(ta.valid.sum()), int(tb.valid.sum()), ""),
    ("predicción coincide", int(agree.sum()), int(agree.sum()), f"de {len(both)} válidos en ambos"),
    ("|d proba| máx", "", "", f"{(both.proba_a.astype(float) - both.proba_b.astype(float)).abs().max():.2e}" if len(both) else ""),
    ("n_samples iguales", int((m.n_samples_a == m.n_samples_b).sum()), "", f"de {len(m)}"),
    ("decision_delay medio (s)", round(ta.decision_delay.mean(), 4), round(tb.decision_delay.mean(), 4),
     f"d(B-A) media {dd.mean():+.4f}, máx {dd.abs().max():.4f}"),
]
print(pd.DataFrame(rows, columns=["métrica", na, nb, "nota"]).to_string(index=False))

# segmentos: ¿recibieron exactamente las mismas muestras?
sa, sb = os.path.join(a.a, "segments.npz"), os.path.join(a.b, "segments.npz")
if os.path.exists(sa) and os.path.exists(sb):
    za, zb = np.load(sa), np.load(sb)
    ka = {round(float(o), 3): k for k, (o, _, _) in enumerate(za["meta"])}
    kb = {round(float(o), 3): k for k, (o, _, _) in enumerate(zb["meta"])}
    same_n = same_x = 0
    max_dx = 0.0
    common = sorted(set(ka) & set(kb))
    for o in common:
        xa, xb = za[f"t{ka[o]}_x"], zb[f"t{kb[o]}_x"]
        if xa.shape == xb.shape:
            same_n += 1
            if len(xa):
                d = float(np.abs(xa.astype(np.float64) - xb.astype(np.float64)).max())
                max_dx = max(max_dx, d)
                same_x += d == 0.0
            else:
                same_x += 1
    print(f"\nSegmentos crudos: {same_n}/{len(common)} con igual forma, {same_x}/{len(common)} idénticos "
          f"(máx |dx| = {max_dx:.3g} uV)")

print("\nTelemetría")
A, B = tel_summary(a.a), tel_summary(a.b)
print(pd.DataFrame({na: A, nb: B}).T.round(5).to_string())
