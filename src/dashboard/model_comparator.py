"""Backend utilities for the Model Comparator dashboard page."""

from __future__ import annotations

import os
import sys
import time
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.config import MODEL_DIR, PREPROCESS_PATH

# Theoretical random-chance loss for 52 classes: ln(52) ≈ 3.95
_RANDOM_LOSS_52 = 3.951
_BROKEN_LOSS_THRESHOLD = 15.0  # > ~4× random-chance loss
_BROKEN_ACCURACY_THRESHOLD = 0.01  # < 1 % accuracy


def get_available_models() -> list[str]:
    """Return sorted list of .keras filenames in MODEL_DIR."""
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".keras"))


def get_model_file_size_mb(name: str) -> float:
    path = os.path.join(MODEL_DIR, name)
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


@dataclass
class ModelComparisonResult:
    name: str
    file_size_mb: float
    param_count: Optional[int]
    loaded: bool
    load_error: Optional[str]

    # Inference metrics — None when no dataset is provided or inference failed
    accuracy: Optional[float]  # 0–1
    loss: Optional[float]
    per_class_accuracy: Optional[dict]  # {int label (0-indexed): float accuracy}
    mean_inference_ms: Optional[float]
    std_inference_ms: Optional[float]
    n_windows_evaluated: int

    # Health diagnostics
    is_broken: bool
    broken_reasons: list = field(default_factory=list)


def _load_dataset_sample(
    dataset_id: str, n_windows: int, seed: int = 42
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load a reproducible random sample from a preprocessed dataset.

    Returns (X, y_0indexed) or (None, None) if dataset is not found.
    Labels are shifted by -1 to match model output indices (as in train.py).
    """
    x_path = os.path.join(PREPROCESS_PATH, dataset_id, "X.npy")
    y_path = os.path.join(PREPROCESS_PATH, dataset_id, "y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None, None

    X_full = np.load(x_path)
    y_full = np.load(y_path) - 1  # shift to 0-indexed (matches train.py)

    n = min(n_windows, len(X_full))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X_full), size=n, replace=False))
    return X_full[idx].astype(np.float32), y_full[idx].astype(np.int32)


ProgressFn = Callable[[str, int, int], None]


def run_model_comparison(
    model_names: list[str],
    dataset_id: Optional[str] = None,
    n_windows: int = 300,
    n_timing_samples: int = 30,
    progress_callback: Optional[ProgressFn] = None,
) -> list[ModelComparisonResult]:
    """Load and optionally evaluate each model.

    Args:
        model_names: List of .keras filenames inside MODEL_DIR.
        dataset_id: Preprocessed dataset ID (e.g. "1") or None for metadata only.
        n_windows: Number of randomly sampled windows for accuracy/loss evaluation.
        n_timing_samples: Number of single-window predict() calls for latency timing.
        progress_callback: Optional callable(label, done, total).

    Returns:
        List of ModelComparisonResult in the same order as model_names.
    """
    import tensorflow as tf

    X, y = None, None
    if dataset_id is not None:
        X, y = _load_dataset_sample(dataset_id, n_windows)

    total = len(model_names)
    results: list[ModelComparisonResult] = []

    for idx, name in enumerate(model_names):
        if progress_callback:
            progress_callback(f"Loading {name}…", idx, total)

        path = os.path.join(MODEL_DIR, name)
        file_size_mb = get_model_file_size_mb(name)

        # ── Load model ──────────────────────────────────────────────────────
        try:
            model = tf.keras.models.load_model(path)
        except Exception as exc:
            results.append(
                ModelComparisonResult(
                    name=name,
                    file_size_mb=file_size_mb,
                    param_count=None,
                    loaded=False,
                    load_error=str(exc),
                    accuracy=None,
                    loss=None,
                    per_class_accuracy=None,
                    mean_inference_ms=None,
                    std_inference_ms=None,
                    n_windows_evaluated=0,
                    is_broken=True,
                    broken_reasons=["Failed to load model"],
                )
            )
            continue

        param_count = model.count_params()

        # ── Metadata-only mode (no dataset) ─────────────────────────────────
        if X is None:
            results.append(
                ModelComparisonResult(
                    name=name,
                    file_size_mb=file_size_mb,
                    param_count=param_count,
                    loaded=True,
                    load_error=None,
                    accuracy=None,
                    loss=None,
                    per_class_accuracy=None,
                    mean_inference_ms=None,
                    std_inference_ms=None,
                    n_windows_evaluated=0,
                    is_broken=False,
                    broken_reasons=[],
                )
            )
            continue

        # ── Batch inference: accuracy + loss ────────────────────────────────
        if progress_callback:
            progress_callback(f"Running inference on {name}…", idx, total)

        try:
            y_pred_probs = model.predict(X, verbose=0, batch_size=64)
        except Exception as exc:
            results.append(
                ModelComparisonResult(
                    name=name,
                    file_size_mb=file_size_mb,
                    param_count=param_count,
                    loaded=True,
                    load_error=None,
                    accuracy=None,
                    loss=None,
                    per_class_accuracy=None,
                    mean_inference_ms=None,
                    std_inference_ms=None,
                    n_windows_evaluated=len(X),
                    is_broken=True,
                    broken_reasons=[f"Inference failed: {exc}"],
                )
            )
            continue

        all_preds = np.argmax(y_pred_probs, axis=1)
        accuracy = float(np.mean(all_preds == y))

        try:
            loss_val: Optional[float] = float(
                tf.keras.losses.sparse_categorical_crossentropy(y, y_pred_probs)
                .numpy()
                .mean()
            )
        except Exception:
            loss_val = float("nan")

        # ── Per-class accuracy ───────────────────────────────────────────────
        per_class: dict[int, float] = {}
        for label in np.unique(y).tolist():
            mask = y == label
            per_class[int(label)] = float(np.mean(all_preds[mask] == label))

        # ── Per-sample latency timing ────────────────────────────────────────
        if progress_callback:
            progress_callback(f"Measuring latency for {name}…", idx, total)

        n_time = min(n_timing_samples, len(X))
        _ = model.predict(X[:1], verbose=0)  # warm-up (not timed)
        times_ms: list[float] = []
        for i in range(n_time):
            sample = X[i : i + 1]
            t0 = time.perf_counter()
            _ = model.predict(sample, verbose=0)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        mean_inference_ms = float(np.mean(times_ms))
        std_inference_ms = float(np.std(times_ms))

        # ── Broken-model detection ───────────────────────────────────────────
        broken_reasons: list[str] = []

        if len(np.unique(all_preds)) == 1:
            broken_reasons.append(
                f"Constant predictions (always class {int(all_preds[0])})"
            )

        if loss_val is not None and math.isnan(loss_val):
            broken_reasons.append("NaN loss")
        elif loss_val is not None and loss_val > _BROKEN_LOSS_THRESHOLD:
            broken_reasons.append(
                f"Loss too high ({loss_val:.2f}; random ≈ {_RANDOM_LOSS_52:.2f})"
            )

        if accuracy < _BROKEN_ACCURACY_THRESHOLD:
            broken_reasons.append(f"Near-zero accuracy ({accuracy * 100:.1f} %)")

        results.append(
            ModelComparisonResult(
                name=name,
                file_size_mb=file_size_mb,
                param_count=param_count,
                loaded=True,
                load_error=None,
                accuracy=accuracy,
                loss=loss_val,
                per_class_accuracy=per_class,
                mean_inference_ms=mean_inference_ms,
                std_inference_ms=std_inference_ms,
                n_windows_evaluated=len(X),
                is_broken=bool(broken_reasons),
                broken_reasons=broken_reasons,
            )
        )

    if progress_callback:
        progress_callback("Done", total, total)

    return results
