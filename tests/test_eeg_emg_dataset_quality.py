"""Tests for EEG–EMG dataset quality checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.dashboard.eeg_emg_dataset_quality import check_eeg_emg_npz


def test_check_eeg_emg_npz_passes_minimal_pair(tmp_path: Path) -> None:
    """Sufficient length and no NaNs should yield windows > 0."""
    t, c_e, c_m = 512, 4, 2
    eeg = np.random.randn(t, c_e).astype(np.float32)
    emg = np.random.randn(t, c_m).astype(np.float32)
    path = tmp_path / "pair.npz"
    np.savez(path, eeg=eeg, emg=emg)

    rep = check_eeg_emg_npz(str(path), window_size=64, step=32)
    assert rep.loaded
    assert rep.n_windows > 0
    assert rep.eeg_key is not None
    assert rep.emg_key is not None
