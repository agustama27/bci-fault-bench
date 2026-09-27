"""Carga de BCI Competition IV 2a (BNCI2014_001) vía MOABB.

Convenciones del dataset en MOABB >= 1.0:
- sesiones: "0train" (entrenamiento) y "1test" (evaluación)
- corridas: "0".."5" (seis corridas de 48 ensayos, 12 por clase)
- eventos: left_hand=1, right_hand=2, feet=3, tongue=4
- intervalo de imaginería usado por MOABB para este dataset: [2, 6] s desde el cue
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import mne
import numpy as np

SFREQ = 250.0
EVENT_ID = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
BINARY_CLASSES = ("left_hand", "right_hand")
TMIN, TMAX = 2.0, 6.0          # ventana de ensayo (4 s) relativa al cue
FMIN, FMAX = 8.0, 30.0         # banda mu + beta
SESSION_TRAIN, SESSION_TEST = "0train", "1test"
RUNS = ["0", "1", "2", "3", "4", "5"]

mne.set_log_level("WARNING")


ARTIFACT_DESC = "bnci_artifact"   # anotación no excluyente que agrega MOABB


def _dataset():
    # MOABB se importa acá para no pagar el costo en módulos que no lo usan.
    # artifact_handling="annotate": los ensayos marcados por los autores del
    # conjunto como contaminados quedan anotados (onset = inicio del ensayo),
    # sin rechazarlos: se reproducen igual, pero no entran al desempeño.
    from moabb.datasets import BNCI2014_001
    return BNCI2014_001(artifact_handling="annotate")


def artifact_mask(raw: mne.io.BaseRaw, events: np.ndarray) -> np.ndarray:
    """True para cada evento cuyo ensayo está marcado con artefacto."""
    sf = raw.info["sfreq"]
    onsets = {int(round(a["onset"] * sf)) + raw.first_samp
              for a in raw.annotations if a["description"] == ARTIFACT_DESC}
    return np.array([int(e[0]) in onsets for e in events], dtype=bool)


def load_subject(subject: int) -> dict:
    """Devuelve {session: {run: Raw}} de un sujeto (descarga si hace falta)."""
    ds = _dataset()
    return ds.get_data(subjects=[subject])[subject]


def eeg_picks(raw: mne.io.BaseRaw) -> list[int]:
    return mne.pick_types(raw.info, eeg=True, eog=False, stim=False)


def events_of(raw: mne.io.BaseRaw) -> np.ndarray:
    """Eventos (n, 3) con códigos EVENT_ID, en el INICIO DEL ENSAYO.

    Gotcha de MOABB (verificado en moabb/datasets/preprocessing.py,
    SetRawAnnotations): las anotaciones del Raw ya están desplazadas
    +interval[0] (= 2 s) respecto del inicio del ensayo y duran 4 s. El canal
    stim 'STI', en cambio, marca el inicio real del ensayo. Se prefiere el
    stim; si solo hay anotaciones, se deshace el desplazamiento.
    """
    stim = mne.pick_types(raw.info, stim=True)
    if len(stim):
        ev = mne.find_events(raw, stim_channel=raw.ch_names[stim[0]],
                             shortest_event=0, verbose=False)
        return ev[np.isin(ev[:, 2], list(EVENT_ID.values()))]
    if len(raw.annotations) > 0:
        ev, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose=False)
        ev[:, 0] -= int(TMIN * raw.info["sfreq"])
        return ev
    raise RuntimeError("La corrida no tiene eventos ni canal stim")


def epochs_of(raw: mne.io.BaseRaw, classes=BINARY_CLASSES,
              tmin=TMIN, tmax=TMAX, fmin=FMIN, fmax=FMAX,
              drop_artifacts=True) -> mne.Epochs:
    """Épocas de las clases pedidas, EEG solamente, banda [fmin, fmax]."""
    ev = events_of(raw)
    keep = ~artifact_mask(raw, ev) if drop_artifacts else np.ones(len(ev), bool)
    raw = raw.copy().pick(eeg_picks(raw))
    raw.filter(fmin, fmax, method="iir", verbose=False)
    ev_id = {c: EVENT_ID[c] for c in classes}
    return mne.Epochs(raw, ev[keep], event_id=ev_id, tmin=tmin, tmax=tmax,
                      baseline=None, preload=True, verbose=False)


@dataclass
class XY:
    X: np.ndarray   # (n_trials, n_channels, n_times)
    y: np.ndarray   # códigos de EVENT_ID
    def __len__(self):
        return len(self.y)


def session_xy(subject_data: dict, session: str, runs=RUNS) -> XY:
    """Concatena las corridas de una sesión en (X, y)."""
    Xs, ys = [], []
    for r in runs:
        ep = epochs_of(subject_data[session][r])
        Xs.append(ep.get_data(copy=False))
        ys.append(ep.events[:, 2])
    return XY(np.concatenate(Xs), np.concatenate(ys))


def data_dir() -> str:
    """Carpeta de descarga de MOABB (respeta MNE_DATA si está definida)."""
    return os.environ.get("MNE_DATA", os.path.join(os.path.expanduser("~"), "mne_data"))
