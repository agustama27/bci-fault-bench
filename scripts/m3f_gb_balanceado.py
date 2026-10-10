"""M3f: gradient boosting con pesos balanceados (2026-10-09).

Revisión de la Entrega 3: en detectores_ap.py, la regresión logística y random forest se entrenan con
class_weight="balanced" y gradient boosting no (GradientBoostingClassifier no tiene class_weight).
Este script repite exactamente la evaluación LOSO de detectores_ap.py para gradient boosting, pero
pasando sample_weight balanceado al ajuste, y reporta las mismas métricas (incluida la precisión promedio).
Uso: python scripts/m3f_gb_balanceado.py --analysis campaign-2026-09-27/analysis
Salida: <analysis>/t_detectores_gb_balanceado.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

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
X = V[S.FEATURES].fillna(0.0).to_numpy(dtype=float)
prev = float(y.mean())
print(f"{len(V)} ventanas, {y.sum()} degradadas, prevalencia {prev:.4f}")

rows = []
for name, weighted in [("gradient_boosting", False), ("gradient_boosting_balanceado", True)]:
    pred = np.zeros(len(y), dtype=int)
    proba = np.zeros(len(y), dtype=float)
    for s in np.unique(subj):
        tr, te = subj != s, subj == s
        if y[tr].min() == y[tr].max():
            continue
        clf = GradientBoostingClassifier(random_state=0)
        if weighted:
            clf.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        else:
            clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
        proba[te] = clf.predict_proba(X[te])[:, 1]
        print(name, "sujeto", s, "ok", flush=True)
    rows.append(dict(detector=name, **S.detection_scores(y, pred.astype(bool), tsec),
                     average_precision=float(average_precision_score(y, proba)), alarm_rate=float(pred.mean())))

t = pd.DataFrame(rows)
t["prevalence"] = prev
t["lift"] = t["precision"] / prev
t.to_csv(os.path.join(a.analysis, "t_detectores_gb_balanceado.csv"), index=False)
print(t.to_string(index=False))
