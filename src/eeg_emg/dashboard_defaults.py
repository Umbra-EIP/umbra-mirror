"""Canonical paths for EEG→EMG dashboard auto-discovery.

Streamlit pages import from here so they still work if ``src.config`` is an older
checkout without ``EEG_EMG_DEFAULT_NPZ`` / ``EEG_EMG_DEFAULT_PTH``.
"""

from __future__ import annotations

import importlib

_cfg = importlib.import_module("src.config")

EEG_EMG_DATA_DIR = _cfg.EEG_EMG_DATA_DIR
EEG_EMG_MODEL_DIR = _cfg.EEG_EMG_MODEL_DIR

EEG_EMG_DEFAULT_NPZ = getattr(
    _cfg,
    "EEG_EMG_DEFAULT_NPZ",
    "data/eeg_emg/dataset_augmented.npz",
)
EEG_EMG_DEFAULT_PTH = getattr(
    _cfg,
    "EEG_EMG_DEFAULT_PTH",
    "src/eeg_emg/eeg2emg_best.pth",
)
EEG_EMG_SUBJECT_MODEL_PTH = getattr(
    _cfg,
    "EEG_EMG_SUBJECT_MODEL_PTH",
    "src/eeg_emg/eeg2emg_single_subject_best.pth",
)
