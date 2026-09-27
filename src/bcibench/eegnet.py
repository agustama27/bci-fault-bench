"""EEGNet (Lawhern et al., 2018) como segundo decodificador, para validez externa.

Arquitectura compacta: conv temporal → conv espacial en profundidad → separable →
clasificador lineal. Parámetros de referencia del artículo para 250 Hz: F1=8, D=2,
F2=16, kernel temporal de 64 muestras (≈ 250 ms), dropout 0,5 (rango intra-sujeto).

Entrada: (n, 22, 1000) en voltios filtrados 8-30 Hz; se estandariza por canal con
las estadísticas del entrenamiento. Ventanas con muestras faltantes se rellenan
con ceros en la grilla nominal (una BCI real debe producir un tensor de tamaño fijo).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(max(1, torch.get_num_threads() // 2))


class EEGNetModel(nn.Module):
    def __init__(self, n_ch=22, n_times=1000, n_classes=2, F1=8, D=2, F2=16, k1=64, drop=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop),
        )
        with torch.no_grad():
            n_feat = self.block2(self.block1(torch.zeros(1, 1, n_ch, n_times))).numel()
        self.fc = nn.Linear(n_feat, n_classes)

    def forward(self, x):                # x: (n, ch, t)
        x = self.block1(x.unsqueeze(1))
        x = self.block2(x)
        return self.fc(x.flatten(1))


@dataclass
class EEGNetDecoder:
    subject: int
    state: dict
    mu: np.ndarray            # (ch, 1) media por canal del entrenamiento
    sd: np.ndarray            # (ch, 1)
    classes: tuple
    n_times: int
    _model: EEGNetModel = None

    def model(self) -> EEGNetModel:
        if self._model is None:
            m = EEGNetModel(n_ch=len(self.mu), n_times=self.n_times, n_classes=len(self.classes))
            m.load_state_dict(self.state)
            m.eval()
            self._model = m
        return self._model

    def _prep(self, X: np.ndarray) -> torch.Tensor:
        Z = (X - self.mu[None]) / self.sd[None]
        return torch.tensor(Z, dtype=torch.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(self.model()(self._prep(X)), dim=1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(self.classes)[self.predict_proba(X).argmax(1)]

    def save(self, path: str):
        d = dict(subject=self.subject, state=self.state, mu=self.mu, sd=self.sd, classes=self.classes, n_times=self.n_times)
        with open(path, "wb") as f:
            pickle.dump(d, f)

    @staticmethod
    def load(path: str) -> "EEGNetDecoder":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return EEGNetDecoder(**d)


def train(subject: int, X: np.ndarray, y: np.ndarray, epochs=300, lr=1e-3, batch=32, seed=0,
          X_val=None, y_val=None, verbose=False) -> EEGNetDecoder:
    """Entrena por sujeto. X en voltios filtrados; y códigos de clase."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    classes = tuple(sorted(np.unique(y)))
    idx = {c: i for i, c in enumerate(classes)}
    mu = X.mean(axis=(0, 2), keepdims=False)[:, None]
    sd = X.std(axis=(0, 2), keepdims=False)[:, None] + 1e-12
    Z = torch.tensor((X - mu[None]) / sd[None], dtype=torch.float32)
    t = torch.tensor([idx[c] for c in y], dtype=torch.long)
    model = EEGNetModel(n_ch=X.shape[1], n_times=X.shape[2], n_classes=len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(Z)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Z[b]), t[b])
            loss.backward()
            opt.step()
        if verbose and X_val is not None and (ep + 1) % 50 == 0:
            model.eval()
            dec = EEGNetDecoder(subject, model.state_dict(), mu, sd, classes, X.shape[2], model)
            acc = (dec.predict(X_val) == y_val).mean()
            print(f"  ep {ep + 1}: loss {loss.item():.3f} val acc {acc:.3f}", flush=True)
    model.eval()
    return EEGNetDecoder(subject=subject, state=model.state_dict(), mu=mu, sd=sd, classes=classes, n_times=X.shape[2])
