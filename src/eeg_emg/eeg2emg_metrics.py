"""Rich metric computation for EEG→EMG regression evaluation.

All metrics operate on numpy arrays shaped ``(N, C, T)``
(windows × channels × time samples) as returned by ``evaluate()``.

Public API
----------
compute_metrics(preds, trues) -> EEGEMGMetrics
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EEGEMGMetrics:
    """All evaluation metrics for an EEG→EMG regression run.

    Global scalars
    --------------
    mse, rmse, mae, r2
        Computed across all windows, channels, and time steps.

    Per-channel lists  (length = n_emg_channels)
    -------------------
    pearson_per_channel
        Pearson r between predicted and true waveforms, averaged over windows.
    r2_per_channel
        R² (coefficient of determination) per channel, averaged over windows.
    mae_per_channel
        Mean absolute error per channel.
    envelope_corr_per_channel
        Pearson r of the signal envelopes (rectified + low-passed via moving
        average) — more informative than raw correlation for burst-like EMG.
    """

    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0

    pearson_per_channel: list[float] = field(default_factory=list)
    r2_per_channel: list[float] = field(default_factory=list)
    mae_per_channel: list[float] = field(default_factory=list)
    envelope_corr_per_channel: list[float] = field(default_factory=list)

    @property
    def mean_pearson(self) -> float:
        """Mean Pearson r across all EMG channels."""
        return float(np.mean(self.pearson_per_channel)) if self.pearson_per_channel else 0.0

    @property
    def n_channels(self) -> int:
        return len(self.pearson_per_channel)


def _pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Pearson r between two 1-D arrays."""
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum())) + eps
    return float((a * b).sum() / denom)


def _r2(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum() + eps
    return float(1.0 - ss_res / ss_tot)


def _envelope(x: np.ndarray, smooth: int = 16) -> np.ndarray:
    """Simple EMG envelope: rectify then moving-average smooth."""
    rect = np.abs(x)
    kernel = np.ones(smooth) / smooth
    return np.convolve(rect, kernel, mode="same")


def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> EEGEMGMetrics:
    """Compute all regression metrics from prediction and ground-truth arrays.

    Parameters
    ----------
    preds, trues:
        Shape ``(N, C, T)`` — windows × EMG channels × time.

    Returns
    -------
    EEGEMGMetrics
    """
    if preds.size == 0 or trues.size == 0:
        return EEGEMGMetrics()

    # Flatten for global scalars
    p_flat = preds.reshape(-1)
    t_flat = trues.reshape(-1)

    mse = float(np.mean((p_flat - t_flat) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p_flat - t_flat)))

    # Global R²
    ss_res = float(np.sum((t_flat - p_flat) ** 2))
    ss_tot = float(np.sum((t_flat - t_flat.mean()) ** 2)) + 1e-8
    r2 = 1.0 - ss_res / ss_tot

    n_channels = preds.shape[1]
    pearson_ch, r2_ch, mae_ch, env_ch = [], [], [], []

    for c in range(n_channels):
        p_c = preds[:, c, :]  # (N, T)
        t_c = trues[:, c, :]

        # Per-channel Pearson: average over windows
        ch_pearson = float(np.mean([_pearson(p_c[i], t_c[i]) for i in range(len(p_c))]))
        pearson_ch.append(ch_pearson)

        # Per-channel R²
        r2_ch.append(_r2(p_c.reshape(-1), t_c.reshape(-1)))

        # Per-channel MAE
        mae_ch.append(float(np.mean(np.abs(p_c - t_c))))

        # Per-channel envelope correlation: average over windows
        env_corr = float(
            np.mean([_pearson(_envelope(p_c[i]), _envelope(t_c[i])) for i in range(len(p_c))])
        )
        env_ch.append(env_corr)

    return EEGEMGMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        pearson_per_channel=pearson_ch,
        r2_per_channel=r2_ch,
        mae_per_channel=mae_ch,
        envelope_corr_per_channel=env_ch,
    )
