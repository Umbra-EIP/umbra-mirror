"""Shared pytest fixtures for the Umbra test suite."""

from pathlib import Path

import numpy as np
import pytest

# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_STEPS = 200
N_CHANNELS = 10
N_CLASSES = 52


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def models_dir() -> Path:
    """Absolute path to the directory that holds the saved .keras model files."""
    return Path(__file__).parent.parent / "src" / "models"


@pytest.fixture
def emg_window() -> np.ndarray:
    """Single synthetic EMG window with shape (1, 200, 10).

    Values are sampled from a realistic signal range (µV order of magnitude)
    so that the Butterworth pre-processing path doesn't degenerate.
    """
    rng = np.random.default_rng(seed=42)
    return rng.uniform(low=-500.0, high=500.0, size=(1, WINDOW_STEPS, N_CHANNELS)).astype(
        np.float32
    )


@pytest.fixture
def emg_batch() -> np.ndarray:
    """Small batch of synthetic EMG windows with shape (8, 200, 10)."""
    rng = np.random.default_rng(seed=0)
    return rng.uniform(low=-500.0, high=500.0, size=(8, WINDOW_STEPS, N_CHANNELS)).astype(
        np.float32
    )


@pytest.fixture(scope="session")
def built_model():
    """Pre-built (untrained) CNN-LSTM model, shared across the session to avoid rebuilding."""
    from src.emg_movement.model import build_cnn_lstm

    return build_cnn_lstm(input_shape=(WINDOW_STEPS, N_CHANNELS), num_classes=N_CLASSES)
