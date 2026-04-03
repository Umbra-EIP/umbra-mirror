"""Tests for group-aware train/val indexing used in single-subject EEG→EMG training."""

from __future__ import annotations

import numpy as np
import pytest

from src.eeg_emg.eeg2emg_train_single_subject import train_val_indices_grouped


def test_grouped_split_no_overlap_between_groups() -> None:
    n, gs = 12, 3
    tr, va = train_val_indices_grouped(n, gs, val_ratio=0.25, seed=0)
    assert len(np.intersect1d(tr, va)) == 0
    assert set(tr.tolist()) | set(va.tolist()) == set(range(n))

    tr_groups = {i // gs for i in tr}
    va_groups = {i // gs for i in va}
    assert len(tr_groups & va_groups) == 0


def test_grouped_split_raises_when_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least 2 windows"):
        train_val_indices_grouped(1, 3, 0.2, seed=0)
