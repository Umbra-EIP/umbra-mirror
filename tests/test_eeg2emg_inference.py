"""Tests for EEG→EMG inference helpers (no training)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eeg_emg.eeg2emg_inference import prepare_eeg_emg_trials


def test_prepare_eeg_emg_trials_from_npz(tmp_path: Path) -> None:
    """Minimal paired EEG/EMG .npz loads to trial format."""
    eeg = np.random.randn(100, 4).astype(np.float32)
    emg = np.random.randn(100, 2).astype(np.float32)
    path = tmp_path / "pair.npz"
    np.savez(path, eeg=eeg, emg=emg)

    e_trial, m_trial, pre_w = prepare_eeg_emg_trials(str(path))
    assert e_trial.ndim == 3
    assert m_trial.ndim == 3
    assert e_trial.shape[0] == m_trial.shape[0]
    assert pre_w is False
