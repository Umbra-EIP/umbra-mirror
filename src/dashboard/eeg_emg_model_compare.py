"""Compare multiple EEG→EMG PyTorch checkpoints on the same `.npz` (regression metrics)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

from src.config import EEG_EMG_COMPARISON_REPORTS_DIR, EEG_EMG_MODEL_DIR
from src.eeg_emg.eeg2emg_inference import run_inference


def _torch_load_cpu(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dataclass
class ModelScoreRow:
    """Per-checkpoint metrics on a fixed protocol."""

    model_file: str
    mse: float
    r2: float
    rmse: float = 0.0
    mae: float = 0.0
    mean_pearson: float = 0.0
    best_val_mse_in_checkpoint: Optional[float] = None
    n_eeg_channels: Optional[int] = None
    n_emg_channels: Optional[int] = None
    window_size: Optional[int] = None
    error: Optional[str] = None


@dataclass
class EegEmgComparisonResult:
    """Aggregated comparison across checkpoints."""

    npz_path: str
    generated_at: str
    seed: int
    step: int
    val_ratio: float
    normalize: bool
    resample_eeg: bool
    no_cuda: bool
    rows: list[ModelScoreRow] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


def _read_best_val_from_ckpt(path: str) -> Optional[float]:
    try:
        ckpt = _torch_load_cpu(path)
        return float(ckpt["best_val_mse"]) if ckpt.get("best_val_mse") is not None else None
    except Exception:
        return None


def run_eeg_emg_comparison(
    npz_path: str,
    model_files: list[str],
    *,
    model_dir: str = EEG_EMG_MODEL_DIR,
    step: int = 128,
    val_ratio: float = 0.2,
    normalize: bool = True,
    seed: int = 42,
    resample_eeg: bool = False,
    eeg_fs: float | None = None,
    emg_fs: float | None = None,
    no_cuda: bool = False,
) -> EegEmgComparisonResult:
    """
    Evaluate each checkpoint on ``npz_path`` using the same inference protocol as the decoder.

    ``model_files`` are basename filenames; each is resolved under ``model_dir``.
    """
    result = EegEmgComparisonResult(
        npz_path=npz_path,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        seed=seed,
        step=step,
        val_ratio=val_ratio,
        normalize=normalize,
        resample_eeg=resample_eeg,
        no_cuda=no_cuda,
        config={
            "model_dir": model_dir,
            "eeg_fs": eeg_fs,
            "emg_fs": emg_fs,
        },
    )

    for name in model_files:
        path = os.path.join(model_dir, name)
        row = ModelScoreRow(model_file=name, mse=0.0, r2=0.0)
        row.best_val_mse_in_checkpoint = _read_best_val_from_ckpt(path)
        if not os.path.isfile(path):
            row.error = f"File not found: {path}"
            result.rows.append(row)
            continue
        try:
            inf = run_inference(
                npz_path,
                path,
                step=step,
                val_ratio=val_ratio,
                normalize=normalize,
                seed=seed,
                resample_eeg=resample_eeg,
                eeg_fs=eeg_fs,
                emg_fs=emg_fs,
                no_cuda=no_cuda,
            )
            row.mse = inf.mse
            row.r2 = inf.r2
            row.rmse = inf.metrics.rmse
            row.mae = inf.metrics.mae
            row.mean_pearson = inf.metrics.mean_pearson
            row.n_eeg_channels = inf.model_cfg.get("n_eeg_channels")
            row.n_emg_channels = inf.model_cfg.get("n_emg_channels")
            row.window_size = inf.model_cfg.get("window_size")
        except Exception as exc:
            row.error = str(exc)
        result.rows.append(row)

    return result


def get_comparison_reports_dir() -> Path:
    p = Path(EEG_EMG_COMPARISON_REPORTS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_comparison(result: EegEmgComparisonResult, name: str = "run") -> Path:
    """Persist comparison as JSON."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = get_comparison_reports_dir() / f"eeg_emg_cmp_{ts}_{safe}.json"
    payload = {
        "name": name,
        "saved_at": result.generated_at,
        "result": asdict(result),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def load_comparison(path: Path) -> Optional[dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def comparison_result_from_saved_payload(payload: dict[str, Any]) -> EegEmgComparisonResult:
    """Rebuild ``EegEmgComparisonResult`` from JSON saved by ``save_comparison``."""
    d = payload["result"]
    return EegEmgComparisonResult(
        npz_path=d["npz_path"],
        generated_at=d["generated_at"],
        seed=d["seed"],
        step=d["step"],
        val_ratio=d["val_ratio"],
        normalize=d["normalize"],
        resample_eeg=d["resample_eeg"],
        no_cuda=d["no_cuda"],
        rows=[ModelScoreRow(**x) for x in d["rows"]],
        config=d.get("config", {}),
    )


def list_saved_comparisons() -> list[tuple[str, str, Path]]:
    """Return (name, saved_at, path) newest first."""
    d = get_comparison_reports_dir()
    if not d.is_dir():
        return []
    out: list[tuple[str, str, Path]] = []
    for p in sorted(d.glob("eeg_emg_cmp_*.json"), reverse=True):
        try:
            with open(p) as f:
                meta = json.load(f)
            out.append((meta.get("name", p.stem), meta.get("saved_at", ""), p))
        except Exception:
            continue
    return out
