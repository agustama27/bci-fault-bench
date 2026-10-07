"""Métricas de detección insensibles al umbral de decisión (versión 3 del Entregable 2).

Reproduce la evaluación de detectores de analyze.py (mismas ventanas, mismos
límites de referencia dejando un sujeto fuera, mismos modelos y semillas) y agrega:
  - prevalencia: proporción de ventanas degradadas = precisión esperable de un
    detector que dispara al azar;
  - lift: precisión / prevalencia (cuántas veces mejor que el azar);
  - average precision (área bajo la curva precisión-sensibilidad), a partir de
    la probabilidad de cada modelo o, para el detector por umbrales, de la
    cantidad de variables fuera de rango.

Uso: python scripts/detectores_ap.py --analysis campaign-2026-09-27/analysis
Salida: <analysis>/t_detectores_ap.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bcibench import stats as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--analysis", required=True)
a = ap.parse_args()

W = pd.read_csv(os.path.join(a.analysis, "windows.csv"))
V = W.dropna(subset=["degraded"]).reset_index(drop=True)
y = V["degraded"].to_numpy(dtype=int)
subj = V["subject"].to_numpy()
tsec = V["sec"].to_numpy(dtype=float)
prev = float(y.mean())
print(f"{len(V)} ventanas, {y.sum()} degradadas, prevalencia {prev:.4f}")

rows = []

# umbrales: alarma = alguna variable fuera de rango; puntaje = cuántas variables fuera de rango
alarm = np.zeros(len(V), dtype=bool)
score = np.zeros(len(V), dtype=float)
for s in np.unique(subj):
    lim = S.reference_limits(V[(V.subject != s) & (V.kind == "none")])
    m = subj == s
    Vm = V[m]
    cnt = np.zeros(m.sum())
    for f, (lo, hi) in lim.items():
        v = Vm[f].to_numpy(dtype=float)
        cnt += (np.nan_to_num(v, nan=lo) < lo) | (np.nan_to_num(v, nan=hi) > hi)
    score[m] = cnt
    alarm[m] = cnt > 0
rows.append(dict(detector="umbrales", **S.detection_scores(y, alarm, tsec),
                 average_precision=float(average_precision_score(y, score))))

# modelos con validación dejando un sujeto fuera, con probabilidades
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

X = V[S.FEATURES].fillna(0.0).to_numpy(dtype=float)
models = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=0),
}
for name, mk in models.items():
    pred = np.zeros(len(y), dtype=int)
    proba = np.zeros(len(y), dtype=float)
    for s in np.unique(subj):
        tr, te = subj != s, subj == s
        if y[tr].min() == y[tr].max():
            continue
        clf = mk().fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
        proba[te] = clf.predict_proba(X[te])[:, 1]
    rows.append(dict(detector=name, **S.detection_scores(y, pred.astype(bool), tsec),
                     average_precision=float(average_precision_score(y, proba))))
    print(name, "ok")

t = pd.DataFrame(rows)
t["prevalence"] = prev
t["lift"] = t["precision"] / prev
t.to_csv(os.path.join(a.analysis, "t_detectores_ap.csv"), index=False)
print(t.to_string(index=False))
