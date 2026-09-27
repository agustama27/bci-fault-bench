"""Decodificador de referencia: CSP + LDA.

Se entrena UNA vez por sujeto con la sesión de entrenamiento (sin fallos,
fuera de línea) y se congela. Métodos: "la primera [sesión] entrena el
decodificador sin fallos y fuera de línea".
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline

N_CSP = 6


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("csp", CSP(n_components=N_CSP, reg=None, log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


@dataclass
class Decoder:
    subject: int
    model: Pipeline
    classes: tuple

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Decoder":
        with open(path, "rb") as f:
            return pickle.load(f)


def train(subject: int, X: np.ndarray, y: np.ndarray) -> Decoder:
    model = make_pipeline()
    model.fit(X, y)
    return Decoder(subject=subject, model=model, classes=tuple(np.unique(y)))


def evaluate(dec: Decoder, X: np.ndarray, y: np.ndarray) -> dict:
    pred = dec.predict(X)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(np.mean(pred == y)),
        "n": int(len(y)),
    }
