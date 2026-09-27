"""Líneas futuras 4 y 5: endurecer EEGNet frente a huecos y decodificador híbrido.

Todo es fuera de línea y reutiliza exactamente la cadena de inferencia de
`scripts/redecode_segments.py`:

    muestras recibidas → grilla nominal [TMIN-PAD, TMAX+PAD) a 250 Hz (1500 muestras)
    → relleno de las posiciones faltantes (cero | interpolación | retención)
    → filtro IIR 8-30 Hz sobre la grilla con relleno → recorte [TMIN, TMAX) (1000)
    → estandarización por canal (mu, sd del entrenamiento) → EEGNet.

Aumento con huecos (entrenamiento): el hueco se aplica en el ESPACIO CRUDO
(muestras en 0 V antes del filtro), igual que en inferencia. Como la señal
entrenada está filtrada 8-30 Hz, mu ≈ 0 y -mu/sd es del orden de 1e-3: el
"silencio" que ve la red es ≈ 0 en el espacio estandarizado, salvo el
transitorio del filtro en los bordes del hueco (que el aumento reproduce).

Canal de máscara (variante eegnet_mask): canal 23 = 1 donde la muestra de la
grilla nominal falta, 0 donde está; mu = 0 y sd = 1 para ese canal, de modo
que pasa la estandarización sin cambios. La máscara NO se filtra.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from mne.filter import filter_data

from . import data
from .eegnet import EEGNetDecoder, EEGNetModel

SF = data.SFREQ
PAD = 1.0
N_WIN = int(round((data.TMAX - data.TMIN) * SF))              # 1000
N_PAD = int(round((data.TMAX - data.TMIN + 2 * PAD) * SF))    # 1500
W0 = int(PAD * SF)                                            # 250: inicio de la ventana en la grilla


# ---------------------------------------------------------------- grilla, relleno y filtro
def grid(ts_rel: np.ndarray, x: np.ndarray, n_ch: int):
    """Coloca las muestras recibidas en la grilla nominal (idéntico a redecode_segments)."""
    g = np.zeros((n_ch, N_PAD), dtype=np.float64)
    idx = np.round((ts_rel - (data.TMIN - PAD)) * SF).astype(int)
    ok = (idx >= 0) & (idx < N_PAD)
    g[:, idx[ok]] = x[ok].T
    present = np.zeros(N_PAD, dtype=bool)
    present[idx[ok]] = True
    return g, present


def fill(g: np.ndarray, present: np.ndarray, how: str) -> np.ndarray:
    """Rellena las posiciones ausentes de la grilla cruda (antes del filtro).

    zero   : 0 V (lo que hace la cadena original).
    interp : interpolación lineal por canal entre la última muestra presente antes
             del hueco y la primera después; en los bordes, el valor más cercano.
    hold   : retención de la última muestra (forward fill); un hueco inicial toma
             el primer valor presente.
    """
    if how == "zero" or present.all() or not present.any():
        return g
    pos = np.flatnonzero(present)
    out = g.copy()
    if how == "interp":
        t = np.arange(g.shape[1])
        miss = ~present
        for c in range(g.shape[0]):
            out[c, miss] = np.interp(t[miss], pos, g[c, pos])   # np.interp: bordes = valor más cercano
        return out
    if how == "hold":
        idx = np.where(present, np.arange(g.shape[1]), 0)
        np.maximum.accumulate(idx, out=idx)
        idx[:pos[0]] = pos[0]
        return g[:, idx]
    raise ValueError(how)


def bandpass(x: np.ndarray) -> np.ndarray:
    """Filtro 8-30 Hz IIR (mismo llamado que en línea). Acepta (..., t); filtra en lote."""
    shp = x.shape
    return filter_data(x.reshape(-1, shp[-1]), SF, data.FMIN, data.FMAX, method="iir",
                       verbose=False).reshape(shp)


def window(gf: np.ndarray) -> np.ndarray:
    return gf[..., W0:W0 + N_WIN]


# ---------------------------------------------------------------- datos crudos de entrenamiento
def raw_padded_epochs(subject_data: dict, session: str, runs=data.RUNS):
    """Épocas CRUDAS (sin filtrar) de [TMIN-PAD, TMAX+PAD) → (n, 22, 1500) en voltios.

    Mismos ensayos que `data.session_xy` (clases binarias, sin artefactos). El valor
    se redondea como viaja por LSL (float32 en µV) para que entrenamiento e
    inferencia vean la misma cuantización.
    """
    import mne
    Xs, ys = [], []
    for r in runs:
        raw = subject_data[session][r]
        ev = data.events_of(raw)
        keep = ~data.artifact_mask(raw, ev)
        raw = raw.copy().pick(data.eeg_picks(raw))
        ev_id = {c: data.EVENT_ID[c] for c in data.BINARY_CLASSES}
        ep = mne.Epochs(raw, ev[keep], event_id=ev_id, tmin=data.TMIN - PAD,
                        tmax=data.TMAX + PAD - 1.0 / SF, baseline=None, preload=True, verbose=False)
        X = ep.get_data(copy=False)[:, :, :N_PAD]
        Xs.append((X * 1e6).astype(np.float32).astype(np.float64) * 1e-6)
        ys.append(ep.events[:, 2])
    return np.concatenate(Xs), np.concatenate(ys)


# ---------------------------------------------------------------- aumento con huecos
def sample_gaps(rng: np.random.Generator, n: int, p: float, lo_s: float, hi_s: float):
    """Para cada ensayo: (inicio, largo) en muestras de la ventana, o (0, 0) si no hay hueco.

    Un único tramo contiguo de largo U(lo_s, hi_s) s, en posición uniforme y
    enteramente dentro de la ventana de 4 s.
    """
    has = rng.random(n) < p
    L = np.round(rng.uniform(lo_s, hi_s, n) * SF).astype(int)
    L = np.minimum(L, N_WIN)
    start = np.array([rng.integers(0, N_WIN - l + 1) for l in L])
    L[~has] = 0
    start[~has] = 0
    return start, L


def apply_gaps(raw_pad: np.ndarray, start: np.ndarray, L: np.ndarray):
    """Borra (0 V) el tramo en la época cruda con relleno; devuelve (crudo, máscara ventana)."""
    X = raw_pad.copy()
    mask = np.zeros((len(X), N_WIN), dtype=np.float32)
    for i in np.flatnonzero(L):
        a, b = W0 + start[i], W0 + start[i] + L[i]
        X[i, :, a:b] = 0.0
        mask[i, start[i]:start[i] + L[i]] = 1.0
    return X, mask


def train_gapaug(subject: int, raw_pad: np.ndarray, y: np.ndarray, p: float = 0.5,
                 lo_s: float = 0.4, hi_s: float = 1.6, use_mask: bool = False,
                 epochs: int = 300, lr: float = 1e-3, batch: int = 32, seed: int = 0,
                 log=None) -> EEGNetDecoder:
    """Mismo entrenamiento que `eegnet.train` (Adam 1e-3, lote 32, 300 épocas, semilla 0)
    con un hueco nuevo por ensayo y por época (probabilidad p).

    La entrada limpia de cada ensayo se filtra una vez; en cada época solo se
    re-filtran los ensayos que recibieron hueco (filtro en lote).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    classes = tuple(sorted(np.unique(y)))
    idx = {c: i for i, c in enumerate(classes)}
    clean = window(bandpass(raw_pad)).astype(np.float64)          # (n, 22, 1000)
    mu = clean.mean(axis=(0, 2))[:, None]
    sd = clean.std(axis=(0, 2))[:, None] + 1e-12
    if use_mask:
        mu = np.vstack([mu, [[0.0]]])
        sd = np.vstack([sd, [[1.0]]])
    n_ch = clean.shape[1] + int(use_mask)
    t = torch.tensor([idx[c] for c in y], dtype=torch.long)
    model = EEGNetModel(n_ch=n_ch, n_times=N_WIN, n_classes=len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(clean)
    zclean = ((clean - mu[None, :clean.shape[1]]) / sd[None, :clean.shape[1]]).astype(np.float32)
    for ep in range(epochs):
        start, L = sample_gaps(rng, n, p, lo_s, hi_s)
        Z = zclean.copy()
        g = np.flatnonzero(L)
        mask = np.zeros((n, N_WIN), dtype=np.float32)
        if len(g):
            Xg, mg = apply_gaps(raw_pad[g], start[g], L[g])
            Z[g] = ((window(bandpass(Xg)) - mu[None, :clean.shape[1]]) / sd[None, :clean.shape[1]]).astype(np.float32)
            mask[g] = mg
        if use_mask:
            Z = np.concatenate([Z, mask[:, None, :]], axis=1)
        Zt = torch.from_numpy(Z)
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Zt[b]), t[b])
            loss.backward()
            opt.step()
        if log is not None and (ep + 1) % 50 == 0:
            log(f"  s{subject:02d} ep {ep + 1}: loss {loss.item():.3f}")
    model.eval()
    return EEGNetDecoder(subject=subject, state=model.state_dict(), mu=mu, sd=sd, classes=classes, n_times=N_WIN)
