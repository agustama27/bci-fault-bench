"""Análisis inferencial (Métodos → Análisis de datos).

RQ1/objetivos 2-3: por tipo de fallo, Friedman sobre {referencia, sev1, sev2, sev3}
con n = sujetos; luego Wilcoxon de cada severidad vs referencia, corrección de
Holm, y tamaño de efecto r = Z / sqrt(n) (rank-biserial apareado como alternativa).
RQ3/objetivo 5: detector por umbrales vs modelos, validación dejando un sujeto fuera.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import FEATURES, KINDS


# ---------------------------------------------------------------- utilidades
def holm(pvals: list[float]) -> list[float]:
    """Corrección de Holm (step-down). Devuelve p ajustados en el orden original."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj.tolist()


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon apareado a vs b con Z aproximado y r = |Z|/sqrt(n)."""
    d = np.asarray(a) - np.asarray(b)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2 or np.all(d == 0):
        return dict(n=n, W=np.nan, p=np.nan, z=np.nan, r=np.nan, median_diff=float(np.median(d)) if n else np.nan)
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
    # rank-biserial apareado: (suma rangos positivos - negativos) / suma total
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    rb = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    z = stats.norm.isf(res.pvalue / 2) * np.sign(rb) if res.pvalue > 0 else np.nan
    return dict(n=n, W=float(res.statistic), p=float(res.pvalue), z=float(z),
                r=float(abs(z) / np.sqrt(n)) if not np.isnan(z) else np.nan,
                rank_biserial=float(rb), median_diff=float(np.median(d)))


# ---------------------------------------------------------------- RQ1: efecto por tipo y severidad
def by_kind_tests(med: pd.DataFrame, metric: str) -> pd.DataFrame:
    """`med`: una fila por (subject, kind, severity, label) con la mediana por sujeto.

    Devuelve una tabla: kind, severity, n, median_ref, median_cond, diff, friedman_chi2,
    friedman_p, wilcoxon_p, p_holm, r, rank_biserial.
    """
    ref = med[med["kind"] == "none"].set_index("subject")[metric]
    rows = []
    for kind in KINDS:
        sub = med[med["kind"] == kind]
        sevs = sorted(sub["severity"].unique())
        if not sevs:
            continue
        # matriz sujetos × (ref + severidades), sujetos completos solamente
        cols = {"ref": ref}
        for s in sevs:
            cols[s] = sub[sub["severity"] == s].set_index("subject")[metric]
        M = pd.DataFrame(cols).dropna()
        if len(M) >= 2 and M.shape[1] >= 3:
            fr = stats.friedmanchisquare(*[M[c].to_numpy() for c in M.columns])
            chi2, fp = float(fr.statistic), float(fr.pvalue)
        else:
            chi2, fp = np.nan, np.nan
        tests = [wilcoxon_paired(M[s].to_numpy(), M["ref"].to_numpy()) for s in sevs]
        padj = holm([t["p"] if not np.isnan(t["p"]) else 1.0 for t in tests])
        for s, t, pa in zip(sevs, tests, padj):
            rows.append(dict(kind=kind, severity=s, n=int(len(M)),
                             median_ref=float(M["ref"].median()), median_cond=float(M[s].median()),
                             diff=float((M[s] - M["ref"]).median()),
                             friedman_chi2=chi2, friedman_p=fp,
                             wilcoxon_p=t["p"], p_holm=pa, r=t["r"], rank_biserial=t.get("rank_biserial", np.nan)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- RQ3: detectores
def threshold_detector(w: pd.DataFrame, ref_limits: dict) -> np.ndarray:
    """Alarma si alguna variable de telemetría sale del rango de la referencia."""
    alarm = np.zeros(len(w), dtype=bool)
    for f, (lo, hi) in ref_limits.items():
        v = w[f].to_numpy(dtype=float)
        alarm |= np.nan_to_num(v, nan=lo) < lo
        alarm |= np.nan_to_num(v, nan=hi) > hi
    return alarm


def reference_limits(w_ref: pd.DataFrame, q_lo=0.5, q_hi=99.5) -> dict:
    lim = {}
    for f in FEATURES:
        v = w_ref[f].dropna().to_numpy(dtype=float)
        if len(v):
            lim[f] = (float(np.percentile(v, q_lo)), float(np.percentile(v, q_hi)))
    return lim


def detection_scores(y_true: np.ndarray, alarm: np.ndarray, t: np.ndarray) -> dict:
    """Precisión, sensibilidad, F1 y retardo de detección (s) respecto del primer
    segundo degradado de cada episodio."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    yt, al = y_true.astype(int), alarm.astype(int)
    out = dict(precision=float(precision_score(yt, al, zero_division=0)),
               recall=float(recall_score(yt, al, zero_division=0)),
               f1=float(f1_score(yt, al, zero_division=0)),
               fpr=float(((al == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1)))
    # retardo: primer alarma dentro de cada episodio de degradación
    delays = []
    in_ep = False
    for i in range(len(yt)):
        if yt[i] == 1 and not in_ep:
            in_ep = True
            j = i
            while j < len(yt) and yt[j] == 1:
                if al[j]:
                    delays.append(float(t[j] - t[i])); break
                j += 1
        elif yt[i] == 0:
            in_ep = False
    out["detection_delay_s"] = float(np.median(delays)) if delays else np.nan
    out["episodes"] = int(sum(1 for i in range(len(yt)) if yt[i] == 1 and (i == 0 or yt[i - 1] == 0)))
    out["episodes_detected"] = len(delays)
    return out


def loso_models(W: pd.DataFrame, subjects: np.ndarray, y: np.ndarray) -> dict:
    """Regresión logística, random forest y gradient boosting con LOSO.

    Hiperparámetros fijos (referencia lineal + dos no lineales); la selección
    interna queda documentada como limitación.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = W[FEATURES].fillna(0.0).to_numpy(dtype=float)
    models = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=0),
    }
    preds = {m: np.zeros(len(y), dtype=int) for m in models}
    for s in np.unique(subjects):
        tr, te = subjects != s, subjects == s
        if y[tr].min() == y[tr].max():
            continue
        for m, mk in models.items():
            clf = mk().fit(X[tr], y[tr])
            preds[m][te] = clf.predict(X[te])
    return preds
