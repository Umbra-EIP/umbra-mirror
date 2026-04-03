"""Inference helpers for EEG→EMG PyTorch checkpoints (dashboard and scripts)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import DataLoader, random_split

from src.eeg_emg.eeg2emg_metrics import EEGEMGMetrics, compute_metrics
from src.eeg_emg.eeg2emg_run import (
    CNNLSTM_EEG2EMG,
    EEGEMGWindowDataset,
    evaluate,
    load_npz_anycase,
    to_trials_format,
)


@dataclass
class InferenceResult:
    """Aggregated metrics and arrays from validation-set inference."""

    mse: float
    r2: float
    preds: np.ndarray
    trues: np.ndarray
    metrics: EEGEMGMetrics = field(default_factory=EEGEMGMetrics)
    model_cfg: dict = field(default_factory=dict)


def prepare_eeg_emg_trials(
    npz_path: str,
    *,
    resample_eeg: bool = False,
    eeg_fs: float | None = None,
    emg_fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Load a .npz and return EEG/EMG trials shaped (N, C, T), matching train script logic.

    Returns
    -------
    eeg_trials, emg_trials, is_pre_windowed
    """
    eeg_raw, emg_raw = load_npz_anycase(npz_path)

    if eeg_raw.ndim == 3 and eeg_raw.shape[1] > eeg_raw.shape[2]:
        eeg_trials = eeg_raw.transpose(0, 2, 1).astype(np.float32)
        emg_trials = emg_raw.transpose(0, 2, 1).astype(np.float32)
        is_pre_windowed = True
    else:
        eeg_trials = to_trials_format(eeg_raw)
        emg_trials = to_trials_format(emg_raw)
        is_pre_windowed = False

    if resample_eeg and eeg_fs and emg_fs and eeg_fs != emg_fs:
        factor = emg_fs / eeg_fs
        newlen = int(np.round(eeg_trials.shape[-1] * factor))
        eeg_trials = np.stack(
            [np.stack([resample(ch, newlen, axis=-1) for ch in trial]) for trial in eeg_trials]
        ).astype(np.float32)

    min_len = min(eeg_trials.shape[-1], emg_trials.shape[-1])
    if eeg_trials.shape[-1] != min_len or emg_trials.shape[-1] != min_len:
        eeg_trials = eeg_trials[..., :min_len]
        emg_trials = emg_trials[..., :min_len]

    return eeg_trials, emg_trials, is_pre_windowed


def _infer_config_from_weights(sd: dict) -> dict:
    """Reconstruct the model config from checkpoint weight shapes.

    Used for checkpoints saved before the ``"config"`` key was introduced.
    """
    # conv1.weight: (cnn_channels, n_eeg_channels, kernel)
    cnn_channels, n_eeg_channels = sd["conv1.weight"].shape[:2]
    # fc.weight: (n_emg_channels, lstm_out)
    n_emg_channels, lstm_out = sd["fc.weight"].shape
    # bidirectional LSTM doubles the output size
    bidirectional = any("_reverse" in k for k in sd)
    lstm_hidden = lstm_out // (2 if bidirectional else 1)
    # count LSTM layers by looking for weight_ih_l{n} keys
    lstm_layers = sum(1 for k in sd if k.startswith("lstm.weight_ih_l") and "_reverse" not in k)
    # window_size cannot be inferred from weights — use training default
    return {
        "n_eeg_channels": int(n_eeg_channels),
        "n_emg_channels": int(n_emg_channels),
        "cnn_channels": int(cnn_channels),
        "lstm_hidden": int(lstm_hidden),
        "lstm_layers": int(lstm_layers),
        "bidirectional": bidirectional,
        "window_size": 256,  # default; override via run_inference(window_size=...)
    }


def load_eeg2emg_model(
    checkpoint_path: str,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[CNNLSTM_EEG2EMG, dict]:
    """Load ``CNNLSTM_EEG2EMG`` weights and config from a ``.pth`` saved by ``eeg2emg_run``.

    Handles both new-style checkpoints (with a ``"config"`` key) and legacy
    checkpoints that contain only ``"model_state_dict"``.
    """
    device = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    # Support both dict-style and bare state-dict checkpoints
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        cfg = ckpt.get("config") or _infer_config_from_weights(sd)
    else:
        # bare state dict saved with torch.save(model.state_dict(), path)
        sd = ckpt
        cfg = _infer_config_from_weights(sd)

    model = CNNLSTM_EEG2EMG(
        cfg["n_eeg_channels"],
        cfg["n_emg_channels"],
        cnn_channels=cfg.get("cnn_channels", 64),
        lstm_hidden=cfg.get("lstm_hidden", 128),
        lstm_layers=cfg.get("lstm_layers", 2),
        bidirectional=cfg.get("bidirectional", False),
    )
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, cfg


def run_inference(
    npz_path: str,
    checkpoint_path: str,
    *,
    window_size: int | None = None,
    step: int = 128,
    val_ratio: float = 0.2,
    normalize: bool = True,
    batch_size: int = 16,
    seed: int = 42,
    resample_eeg: bool = False,
    eeg_fs: float | None = None,
    emg_fs: float | None = None,
    no_cuda: bool = False,
) -> InferenceResult:
    """
    Build windows from ``npz_path``, split train/val like training, and evaluate on val.

    ``window_size`` defaults to checkpoint config value if omitted.
    """
    eeg_trials, emg_trials, is_pre_windowed = prepare_eeg_emg_trials(
        npz_path,
        resample_eeg=resample_eeg,
        eeg_fs=eeg_fs,
        emg_fs=emg_fs,
    )

    device = torch.device("cpu" if no_cuda else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_eeg2emg_model(checkpoint_path, map_location=device)

    ws = window_size if window_size is not None else int(cfg.get("window_size", 256))

    dataset = EEGEMGWindowDataset(
        eeg_trials,
        emg_trials,
        window_size=ws,
        step=step,
        normalize=normalize,
        pre_windowed=is_pre_windowed,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "No windows created: check window_size/step relative to recording length."
        )

    n_total = len(dataset)
    if n_total == 1:
        val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        if n_train < 1:
            n_train, n_val = n_total - 1, 1
        gen = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    preds, trues = evaluate(model, val_loader, device)
    rich = compute_metrics(preds, trues)
    return InferenceResult(
        mse=rich.mse,
        r2=rich.r2,
        preds=preds,
        trues=trues,
        metrics=rich,
        model_cfg=cfg,
    )
