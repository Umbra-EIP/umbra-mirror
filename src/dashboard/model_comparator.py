"""Backend utilities for the Model Comparator dashboard page."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.config import COMPARISON_REPORTS_DIR, MODEL_DIR, PREPROCESS_PATH

# Theoretical random-chance loss for 52 classes: ln(52) ≈ 3.95
_RANDOM_LOSS_52 = 3.951
_BROKEN_LOSS_THRESHOLD = 15.0
_BROKEN_ACCURACY_THRESHOLD = 0.01


# ── Public helpers ────────────────────────────────────────────────────────────


def get_available_models() -> list[str]:
    """Return sorted list of .keras filenames in MODEL_DIR."""
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".keras"))


def get_model_file_size_mb(name: str) -> float:
    try:
        return os.path.getsize(os.path.join(MODEL_DIR, name)) / (1024 * 1024)
    except OSError:
        return 0.0


# ── Private compute helpers ───────────────────────────────────────────────────


def _file_modified_iso(name: str) -> Optional[str]:
    try:
        ts = os.path.getmtime(os.path.join(MODEL_DIR, name))
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return None


def _layer_summary(model) -> list[dict]:
    rows = []
    for layer in model.layers:
        try:
            out_shape = str(layer.output_shape)
        except Exception:
            out_shape = "?"
        rows.append(
            {
                "name": layer.name,
                "type": type(layer).__name__,
                "output_shape": out_shape,
                "params": int(layer.count_params()),
                "trainable": bool(layer.trainable),
            }
        )
    return rows


def _get_model_input_shape(model) -> Optional[tuple]:
    try:
        return tuple(int(d) if d is not None else -1 for d in model.input_shape[1:])
    except Exception:
        return None


def _shapes_compatible(model_shape: Optional[tuple], data_shape: tuple) -> Optional[bool]:
    if model_shape is None:
        return None
    if len(model_shape) != len(data_shape):
        return False
    return all(m == -1 or m == d for m, d in zip(model_shape, data_shape))


def _topk_accuracy(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int) -> float:
    k = min(k, y_pred_probs.shape[1])
    top_k = np.argsort(y_pred_probs, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in top_k[i] for i in range(len(y_true))]))


def _per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict:
    """Compute per-class precision, recall, F1 without sklearn."""
    prec_d: dict[int, float] = {}
    rec_d: dict[int, float] = {}
    f1_d: dict[int, float] = {}
    for lbl in labels:
        tp = int(np.sum((y_pred == lbl) & (y_true == lbl)))
        fp = int(np.sum((y_pred == lbl) & (y_true != lbl)))
        fn = int(np.sum((y_pred != lbl) & (y_true == lbl)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        prec_d[lbl], rec_d[lbl], f1_d[lbl] = p, r, f
    return {"precision": prec_d, "recall": rec_d, "f1": f1_d}


def _compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]
) -> list[list[int]]:
    """Return list-of-lists confusion matrix indexed by `labels` order."""
    idx_map = {lbl: i for i, lbl in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for true, pred in zip(y_true.tolist(), y_pred.tolist()):
        if true in idx_map and pred in idx_map:
            cm[idx_map[true]][idx_map[pred]] += 1
    return cm


def _confidence_and_entropy(y_pred_probs: np.ndarray) -> dict:
    confidences = np.max(y_pred_probs, axis=1)
    eps = 1e-10
    entropy = -np.sum(y_pred_probs * np.log(y_pred_probs + eps), axis=1)
    return {
        "mean_confidence": float(np.mean(confidences)),
        "std_confidence": float(np.std(confidences)),
        "mean_entropy": float(np.mean(entropy)),
        "confidence_distribution": confidences.tolist(),
    }


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class ModelComparisonResult:
    # Core metadata
    name: str
    file_size_mb: float
    file_modified: Optional[str]
    param_count: Optional[int]
    layer_summary: Optional[list]  # list[dict] per layer
    loaded: bool
    load_error: Optional[str]

    # Shape
    model_input_shape: Optional[tuple]  # e.g. (200, 10); -1 means None/dynamic
    dataset_input_shape: Optional[tuple]
    shape_compatible: Optional[bool]

    # Aggregate inference metrics (None if no dataset or inference failed)
    accuracy: Optional[float]
    top3_accuracy: Optional[float]
    top5_accuracy: Optional[float]
    loss: Optional[float]
    macro_f1: Optional[float]

    # Per-class metrics  {int label: float}
    per_class_accuracy: Optional[dict]
    per_class_precision: Optional[dict]
    per_class_recall: Optional[dict]
    per_class_f1: Optional[dict]

    # Confusion matrix (list-of-lists[int], rows=true, cols=predicted, index=class_labels)
    confusion_matrix: Optional[list]
    class_labels: Optional[list]  # sorted int labels present in evaluation

    # Confidence & entropy
    mean_confidence: Optional[float]
    std_confidence: Optional[float]
    mean_entropy: Optional[float]
    confidence_distribution: Optional[list]  # max-softmax per sample

    # Timing
    mean_inference_ms: Optional[float]
    std_inference_ms: Optional[float]
    n_windows_evaluated: int

    # Health
    is_broken: bool
    broken_reasons: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> ModelComparisonResult:
        """Reconstruct from a JSON-loaded dict (handles key/type conversions)."""
        d = dict(d)  # shallow copy to avoid mutating caller's data
        for fld in [
            "per_class_accuracy",
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
        ]:
            if d.get(fld) is not None:
                d[fld] = {int(k): float(v) for k, v in d[fld].items()}
        for fld in ["model_input_shape", "dataset_input_shape"]:
            if d.get(fld) is not None:
                d[fld] = tuple(d[fld])
        known = {f.name for f in fields(cls)}
        cleaned: dict = {k: v for k, v in d.items() if k in known}
        for f in fields(cls):
            if f.name not in cleaned:
                if f.default is not MISSING:
                    cleaned[f.name] = f.default
                elif f.default_factory is not MISSING:
                    cleaned[f.name] = f.default_factory()
                else:
                    cleaned[f.name] = None
        return cls(**cleaned)


# ── Dataset loader ────────────────────────────────────────────────────────────


def _load_dataset_sample(
    dataset_id: str, n_windows: int, seed: int = 42
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load a reproducible random sample. Labels shifted to 0-indexed (matches train.py)."""
    x_path = os.path.join(PREPROCESS_PATH, dataset_id, "X.npy")
    y_path = os.path.join(PREPROCESS_PATH, dataset_id, "y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None, None
    X_full = np.load(x_path)
    y_full = np.load(y_path) - 1
    n = min(n_windows, len(X_full))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X_full), size=n, replace=False))
    return X_full[idx].astype(np.float32), y_full[idx].astype(np.int32)


# ── Main comparison function ──────────────────────────────────────────────────

ProgressFn = Callable[[str, int, int], None]


def _make_incomplete_result(
    name: str,
    file_size_mb: float,
    file_modified: Optional[str],
    param_count: Optional[int],
    layer_summary: Optional[list],
    loaded: bool,
    load_error: Optional[str],
    model_input_shape: Optional[tuple],
    dataset_input_shape: Optional[tuple],
    shape_compatible: Optional[bool],
    n_windows_evaluated: int,
    is_broken: bool,
    broken_reasons: list,
) -> ModelComparisonResult:
    """Build a result with all inference fields set to None."""
    return ModelComparisonResult(
        name=name,
        file_size_mb=file_size_mb,
        file_modified=file_modified,
        param_count=param_count,
        layer_summary=layer_summary,
        loaded=loaded,
        load_error=load_error,
        model_input_shape=model_input_shape,
        dataset_input_shape=dataset_input_shape,
        shape_compatible=shape_compatible,
        accuracy=None,
        top3_accuracy=None,
        top5_accuracy=None,
        loss=None,
        macro_f1=None,
        per_class_accuracy=None,
        per_class_precision=None,
        per_class_recall=None,
        per_class_f1=None,
        confusion_matrix=None,
        class_labels=None,
        mean_confidence=None,
        std_confidence=None,
        mean_entropy=None,
        confidence_distribution=None,
        mean_inference_ms=None,
        std_inference_ms=None,
        n_windows_evaluated=n_windows_evaluated,
        is_broken=is_broken,
        broken_reasons=broken_reasons,
    )


def run_model_comparison(
    model_names: list[str],
    dataset_id: Optional[str] = None,
    n_windows: int = 300,
    n_timing_samples: int = 30,
    progress_callback: Optional[ProgressFn] = None,
) -> list[ModelComparisonResult]:
    """Load and optionally evaluate each model.

    Args:
        model_names: .keras filenames inside MODEL_DIR.
        dataset_id: Preprocessed dataset ID or None for metadata-only.
        n_windows: Windows randomly sampled (seed 42) for accuracy/loss.
        n_timing_samples: Single-window predict() calls for latency.
        progress_callback: Optional callable(label, done, total).
    """
    import tensorflow as tf

    X, y = None, None
    if dataset_id is not None:
        X, y = _load_dataset_sample(dataset_id, n_windows)

    dataset_input_shape = tuple(X.shape[1:]) if X is not None else None
    total = len(model_names)
    results: list[ModelComparisonResult] = []

    for idx, name in enumerate(model_names):
        if progress_callback:
            progress_callback(f"Loading {name}…", idx, total)

        file_size_mb = get_model_file_size_mb(name)
        file_modified = _file_modified_iso(name)
        path = os.path.join(MODEL_DIR, name)

        # ── Load ──────────────────────────────────────────────────────────────
        try:
            model = tf.keras.models.load_model(path)
        except Exception as exc:
            results.append(
                _make_incomplete_result(
                    name=name,
                    file_size_mb=file_size_mb,
                    file_modified=file_modified,
                    param_count=None,
                    layer_summary=None,
                    loaded=False,
                    load_error=str(exc),
                    model_input_shape=None,
                    dataset_input_shape=dataset_input_shape,
                    shape_compatible=None,
                    n_windows_evaluated=0,
                    is_broken=True,
                    broken_reasons=["Failed to load model"],
                )
            )
            continue

        param_count = model.count_params()
        model_input_shape = _get_model_input_shape(model)
        layer_sum = _layer_summary(model)
        shape_ok = (
            _shapes_compatible(model_input_shape, dataset_input_shape)
            if dataset_input_shape is not None
            else None
        )

        # ── Metadata-only mode ────────────────────────────────────────────────
        if X is None:
            results.append(
                _make_incomplete_result(
                    name=name,
                    file_size_mb=file_size_mb,
                    file_modified=file_modified,
                    param_count=param_count,
                    layer_summary=layer_sum,
                    loaded=True,
                    load_error=None,
                    model_input_shape=model_input_shape,
                    dataset_input_shape=None,
                    shape_compatible=None,
                    n_windows_evaluated=0,
                    is_broken=False,
                    broken_reasons=[],
                )
            )
            continue

        # ── Input shape incompatibility ───────────────────────────────────────
        if shape_ok is False:
            results.append(
                _make_incomplete_result(
                    name=name,
                    file_size_mb=file_size_mb,
                    file_modified=file_modified,
                    param_count=param_count,
                    layer_summary=layer_sum,
                    loaded=True,
                    load_error=None,
                    model_input_shape=model_input_shape,
                    dataset_input_shape=dataset_input_shape,
                    shape_compatible=False,
                    n_windows_evaluated=0,
                    is_broken=True,
                    broken_reasons=[
                        f"Input shape mismatch: model expects {model_input_shape}, "
                        f"dataset provides {dataset_input_shape}"
                    ],
                )
            )
            continue

        # ── Batch inference ───────────────────────────────────────────────────
        if progress_callback:
            progress_callback(f"Running inference on {name}…", idx, total)

        try:
            y_pred_probs = model.predict(X, verbose=0, batch_size=64)
        except Exception as exc:
            results.append(
                _make_incomplete_result(
                    name=name,
                    file_size_mb=file_size_mb,
                    file_modified=file_modified,
                    param_count=param_count,
                    layer_summary=layer_sum,
                    loaded=True,
                    load_error=None,
                    model_input_shape=model_input_shape,
                    dataset_input_shape=dataset_input_shape,
                    shape_compatible=shape_ok,
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
                tf.keras.losses.sparse_categorical_crossentropy(y, y_pred_probs).numpy().mean()
            )
        except Exception:
            loss_val = float("nan")

        # ── Top-K ─────────────────────────────────────────────────────────────
        top3 = _topk_accuracy(y, y_pred_probs, 3)
        top5 = _topk_accuracy(y, y_pred_probs, 5)

        # ── Per-class metrics ─────────────────────────────────────────────────
        class_labels = sorted(np.unique(y).tolist())
        pcm = _per_class_metrics(y, all_preds, class_labels)
        per_class_accuracy = {
            lbl: float(np.mean(all_preds[y == lbl] == lbl)) for lbl in class_labels
        }
        macro_f1 = float(np.mean(list(pcm["f1"].values()))) if pcm["f1"] else None

        # ── Confusion matrix ──────────────────────────────────────────────────
        cm = _compute_confusion_matrix(y, all_preds, class_labels)

        # ── Confidence & entropy ──────────────────────────────────────────────
        conf_ent = _confidence_and_entropy(y_pred_probs)

        # ── Per-sample latency ────────────────────────────────────────────────
        if progress_callback:
            progress_callback(f"Measuring latency for {name}…", idx, total)

        n_time = min(n_timing_samples, len(X))
        _ = model.predict(X[:1], verbose=0)  # warm-up
        times_ms: list[float] = []
        for i in range(n_time):
            t0 = time.perf_counter()
            _ = model.predict(X[i : i + 1], verbose=0)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(times_ms))
        std_ms = float(np.std(times_ms))

        # ── Broken detection ──────────────────────────────────────────────────
        broken: list[str] = []
        if len(np.unique(all_preds)) == 1:
            broken.append(f"Constant predictions (always class {int(all_preds[0])})")
        if loss_val is not None and math.isnan(loss_val):
            broken.append("NaN loss")
        elif loss_val is not None and loss_val > _BROKEN_LOSS_THRESHOLD:
            broken.append(f"Loss too high ({loss_val:.2f}; random ≈ {_RANDOM_LOSS_52:.2f})")
        if accuracy < _BROKEN_ACCURACY_THRESHOLD:
            broken.append(f"Near-zero accuracy ({accuracy * 100:.1f} %)")

        results.append(
            ModelComparisonResult(
                name=name,
                file_size_mb=file_size_mb,
                file_modified=file_modified,
                param_count=param_count,
                layer_summary=layer_sum,
                loaded=True,
                load_error=None,
                model_input_shape=model_input_shape,
                dataset_input_shape=dataset_input_shape,
                shape_compatible=shape_ok,
                accuracy=accuracy,
                top3_accuracy=top3,
                top5_accuracy=top5,
                loss=loss_val,
                macro_f1=macro_f1,
                per_class_accuracy=per_class_accuracy,
                per_class_precision=pcm["precision"],
                per_class_recall=pcm["recall"],
                per_class_f1=pcm["f1"],
                confusion_matrix=cm,
                class_labels=class_labels,
                mean_confidence=conf_ent["mean_confidence"],
                std_confidence=conf_ent["std_confidence"],
                mean_entropy=conf_ent["mean_entropy"],
                confidence_distribution=conf_ent["confidence_distribution"],
                mean_inference_ms=mean_ms,
                std_inference_ms=std_ms,
                n_windows_evaluated=len(X),
                is_broken=bool(broken),
                broken_reasons=broken,
            )
        )

    if progress_callback:
        progress_callback("Done", total, total)

    return results


# ── Persistence ───────────────────────────────────────────────────────────────


def _result_to_dict(r: ModelComparisonResult) -> dict:
    d = dataclasses.asdict(r)
    # JSON requires string dict keys
    for fld in [
        "per_class_accuracy",
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
    ]:
        if d.get(fld) is not None:
            d[fld] = {str(k): v for k, v in d[fld].items()}
    return d


def save_comparison(
    results: list[ModelComparisonResult],
    config: dict,
    name: str = "default",
) -> Path:
    os.makedirs(COMPARISON_REPORTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(COMPARISON_REPORTS_DIR) / f"comparison_{ts}_{name}.json"
    payload = {
        "name": name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "results": [_result_to_dict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_comparison(path: Path) -> tuple[list[ModelComparisonResult], dict]:
    with open(path) as f:
        payload = json.load(f)
    results = [ModelComparisonResult.from_dict(d) for d in payload["results"]]
    return results, payload.get("config", {})


def list_saved_comparisons() -> list[tuple[str, str, Path]]:
    """Return list of (name, saved_at, path) sorted newest first."""
    dir_path = Path(COMPARISON_REPORTS_DIR)
    if not dir_path.is_dir():
        return []
    items = []
    for p in sorted(dir_path.glob("comparison_*.json"), reverse=True):
        try:
            with open(p) as f:
                meta = json.load(f)
            items.append((meta.get("name", p.stem), meta.get("saved_at", ""), p))
        except Exception:
            continue
    return items
