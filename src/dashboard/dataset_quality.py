# src/dashboard/dataset_quality.py
"""Dataset quality checks for preprocessed EMG datasets (X.npy / y.npy)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import PREPROCESS_PATH, QUALITY_REPORTS_DIR, REST_LABEL

# Thresholds for pass/fail and warnings
MIN_SAMPLES_PER_CLASS_TRAIN = 10
MIN_SAMPLES_PER_CLASS_VAL = 2
MIN_SAMPLES_PER_CLASS_TEST = 2
MIN_CLASS_BALANCE_RATIO = 0.05  # min count / max count per class
VARIANCE_THRESHOLD_FLAT = 1e-10  # windows with per-channel var below this are "flat"


@dataclass
class QualityThresholds:
    """Optional thresholds for quality checks. None means use module default."""

    min_samples_train: Optional[int] = None
    min_samples_val: Optional[int] = None
    min_samples_test: Optional[int] = None
    min_balance_ratio: Optional[float] = None


def get_available_datasets() -> list[str]:
    """Return sorted list of dataset IDs (folder names) in PREPROCESS_PATH."""
    if not os.path.isdir(PREPROCESS_PATH):
        return []
    ids = [
        d
        for d in os.listdir(PREPROCESS_PATH)
        if os.path.isdir(os.path.join(PREPROCESS_PATH, d)) and d.isdigit()
    ]
    return sorted(ids, key=int)


def load_dataset(dataset_id: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load X.npy and y.npy for a dataset. Returns (X, y) or (None, None) on error."""
    folder = os.path.join(PREPROCESS_PATH, str(dataset_id))
    x_path = os.path.join(folder, "X.npy")
    y_path = os.path.join(folder, "y.npy")
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        return None, None
    try:
        X = np.load(x_path)
        y = np.load(y_path)
        return X, y
    except Exception:
        return None, None


@dataclass
class QualityReport:
    """Result of running quality checks on a dataset."""

    dataset_id: str
    loaded: bool
    error: Optional[str] = None

    # Report identity
    report_name: str = "default"
    thresholds_used: Optional[dict] = None  # resolved thresholds that were applied

    # Basic stats (when loaded)
    x_shape: Optional[tuple] = None
    y_shape: Optional[tuple] = None
    x_dtype: Optional[str] = None
    y_dtype: Optional[str] = None

    # X value range (global; for scaling/clipping sanity)
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    x_mean: Optional[float] = None
    x_mean_per_channel: Optional[list] = None  # when X has channel dim

    # Integrity
    x_nan_count: int = 0
    x_inf_count: int = 0
    y_nan_count: int = 0
    y_inf_count: int = 0

    # Labels
    unique_labels: Optional[np.ndarray] = None
    num_classes: int = 0
    rest_label_count: int = 0
    counts_per_class: Optional[dict] = None
    min_count: int = 0
    max_count: int = 0
    balance_ratio: float = 0.0

    # Split simulation (70/15/15)
    train_per_class: Optional[dict] = None
    val_per_class: Optional[dict] = None
    test_per_class: Optional[dict] = None
    classes_with_few_train: list = field(default_factory=list)
    classes_with_few_val: list = field(default_factory=list)
    classes_with_few_test: list = field(default_factory=list)

    # Suspicious windows
    flat_window_count: int = 0
    zero_window_count: int = 0
    total_windows: int = 0

    # Verdict
    passed: bool = False
    warnings: list[str] = field(default_factory=list)


def _report_to_dict(report: QualityReport) -> dict[str, Any]:
    """Convert QualityReport to a JSON-serializable dict."""
    d = asdict(report)
    if report.unique_labels is not None:
        d["unique_labels"] = report.unique_labels.tolist()
    if d.get("x_shape") is not None:
        d["x_shape"] = list(d["x_shape"])
    if d.get("y_shape") is not None:
        d["y_shape"] = list(d["y_shape"])
    for key in (
        "counts_per_class",
        "train_per_class",
        "val_per_class",
        "test_per_class",
    ):
        if d.get(key) is not None:
            d[key] = {str(k): v for k, v in d[key].items()}

    # Ensure no numpy scalars (JSON does not serialize np.int64 etc.)
    def _to_native(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(x) for x in obj]
        return obj

    return _to_native(d)


def _report_from_dict(d: dict[str, Any]) -> QualityReport:
    """Build QualityReport from a dict (e.g. loaded from JSON)."""
    d = dict(d)
    for key in (
        "counts_per_class",
        "train_per_class",
        "val_per_class",
        "test_per_class",
    ):
        if d.get(key) is not None:
            d[key] = {int(k): v for k, v in d[key].items()}
    if d.get("unique_labels") is not None:
        d["unique_labels"] = np.array(d["unique_labels"])
    if d.get("x_shape") is not None:
        d["x_shape"] = tuple(d["x_shape"])
    if d.get("y_shape") is not None:
        d["y_shape"] = tuple(d["y_shape"])
    return QualityReport(**d)


def get_reports_dir() -> Path:
    """Return the directory for saved quality reports; create if needed."""
    path = Path(QUALITY_REPORTS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _report_filename(dataset_id: str, report_name: str) -> str:
    """Return the filename for a report given dataset_id and report_name."""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in report_name)
    return f"dataset_{dataset_id}_{safe_name}.json"


def list_saved_reports() -> list[tuple[str, str, Path]]:
    """
    List saved reports: [(dataset_id, report_name, path), ...] sorted by (dataset_id, report_name).
    Filenames are expected to be dataset_<id>_<name>.json.
    """
    reports_dir = get_reports_dir()
    out = []
    for f in reports_dir.glob("dataset_*.json"):
        try:
            stem = f.stem  # e.g. "dataset_1_default" or "dataset_1_strict"
            parts = stem.split("_", 2)  # ["dataset", "<id>", "<name>"]
            if len(parts) < 3:
                continue
            dataset_id = parts[1]
            report_name = parts[2]
            if dataset_id.isdigit():
                out.append((dataset_id, report_name, f))
        except Exception:
            continue
    return sorted(out, key=lambda x: (int(x[0]), x[1]))


def save_report(report: QualityReport) -> Path:
    """Save a QualityReport to QUALITY_REPORTS_DIR using dataset_id + report_name. Returns path."""
    reports_dir = get_reports_dir()
    filename = _report_filename(report.dataset_id, report.report_name)
    path = reports_dir / filename
    with open(path, "w") as f:
        json.dump(_report_to_dict(report), f, indent=2)
    return path


def load_report(path: Path) -> Optional[QualityReport]:
    """Load a QualityReport from a JSON file. Returns None on error."""
    try:
        with open(path) as f:
            d = json.load(f)
        return _report_from_dict(d)
    except Exception:
        return None


def _stratified_split_indices(
    y: np.ndarray, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices for train/val/test with stratified split."""
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = np.arange(n)
    rng.shuffle(indices)
    y_shuffled = y[indices]

    # Approximate stratified split by assigning each class proportionally
    train_idx, val_idx, test_idx = [], [], []
    for label in np.unique(y_shuffled):
        mask = y_shuffled == label
        class_indices = indices[mask]
        n_class = len(class_indices)
        n_train = max(1, int(n_class * train_frac))
        n_val = max(0, int(n_class * val_frac))
        n_test = n_class - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n_class - n_train
        train_idx.extend(class_indices[:n_train].tolist())
        val_idx.extend(class_indices[n_train : n_train + n_val].tolist())
        test_idx.extend(class_indices[n_train + n_val :].tolist())

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def check_quality(
    dataset_id: str,
    thresholds: Optional[QualityThresholds] = None,
    report_name: str = "default",
) -> QualityReport:
    """
    Run all quality checks on a preprocessed dataset.
    Returns a QualityReport with metrics and pass/fail.
    Optional thresholds override the default min samples and balance ratio.
    report_name is stored in the report and used as part of the saved filename.
    """
    report = QualityReport(dataset_id=dataset_id, loaded=False, report_name=report_name)
    t = thresholds or QualityThresholds()
    min_train = (
        t.min_samples_train
        if t.min_samples_train is not None
        else MIN_SAMPLES_PER_CLASS_TRAIN
    )
    min_val = (
        t.min_samples_val
        if t.min_samples_val is not None
        else MIN_SAMPLES_PER_CLASS_VAL
    )
    min_test = (
        t.min_samples_test
        if t.min_samples_test is not None
        else MIN_SAMPLES_PER_CLASS_TEST
    )
    min_balance = (
        t.min_balance_ratio
        if t.min_balance_ratio is not None
        else MIN_CLASS_BALANCE_RATIO
    )

    report.thresholds_used = {
        "min_samples_train": min_train,
        "min_samples_val": min_val,
        "min_samples_test": min_test,
        "min_balance_ratio": min_balance,
    }

    X, y = load_dataset(dataset_id)
    if X is None or y is None:
        report.error = "Failed to load X.npy or y.npy"
        return report

    report.loaded = True
    report.x_shape = X.shape
    report.y_shape = y.shape
    report.x_dtype = str(X.dtype)
    report.y_dtype = str(y.dtype)
    report.total_windows = len(y)

    # X value range (global and per-channel when applicable)
    if np.issubdtype(X.dtype, np.floating) or np.issubdtype(X.dtype, np.integer):
        report.x_min = float(np.nanmin(X))
        report.x_max = float(np.nanmax(X))
        report.x_mean = float(np.nanmean(X))
        if X.ndim == 3:
            # (n_windows, time, channels) -> mean per channel
            report.x_mean_per_channel = np.nanmean(X, axis=(0, 1)).tolist()

    # Integrity
    report.x_nan_count = int(np.isnan(X).sum())
    report.x_inf_count = int(np.isinf(X).sum())
    report.y_nan_count = int(np.isnan(y).sum())
    report.y_inf_count = int(np.isinf(y).sum())

    if report.x_nan_count > 0 or report.x_inf_count > 0:
        report.warnings.append(
            f"X contains invalid values: {report.x_nan_count} NaN, {report.x_inf_count} Inf"
        )
    if report.y_nan_count > 0 or report.y_inf_count > 0:
        report.warnings.append(
            f"y contains invalid values: {report.y_nan_count} NaN, {report.y_inf_count} Inf"
        )

    # Labels
    report.unique_labels = np.unique(y)
    report.num_classes = len(report.unique_labels)
    report.rest_label_count = int((y == REST_LABEL).sum())
    counts = {}
    for lab in report.unique_labels:
        counts[int(lab)] = int((y == lab).sum())
    report.counts_per_class = counts
    report.min_count = min(counts.values()) if counts else 0
    report.max_count = max(counts.values()) if counts else 0
    report.balance_ratio = (
        report.min_count / report.max_count if report.max_count > 0 else 0.0
    )

    if report.balance_ratio < min_balance:
        report.warnings.append(
            f"Strong class imbalance: ratio {report.balance_ratio:.3f} "
            f"(min {report.min_count} / max {report.max_count})"
        )

    # Split simulation
    try:
        train_idx, val_idx, test_idx = _stratified_split_indices(y)
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        report.train_per_class = {
            int(lab): int((y_train == lab).sum()) for lab in report.unique_labels
        }
        report.val_per_class = {
            int(lab): int((y_val == lab).sum()) for lab in report.unique_labels
        }
        report.test_per_class = {
            int(lab): int((y_test == lab).sum()) for lab in report.unique_labels
        }
        for lab in report.unique_labels:
            if report.train_per_class[int(lab)] < min_train:
                report.classes_with_few_train.append(int(lab))
            if report.val_per_class[int(lab)] < min_val:
                report.classes_with_few_val.append(int(lab))
            if report.test_per_class[int(lab)] < min_test:
                report.classes_with_few_test.append(int(lab))
        if report.classes_with_few_train:
            report.warnings.append(
                f"Classes with < {min_train} samples in train: "
                f"{report.classes_with_few_train}"
            )
        if report.classes_with_few_val:
            report.warnings.append(
                f"Classes with < {min_val} samples in val: "
                f"{report.classes_with_few_val}"
            )
        if report.classes_with_few_test:
            report.warnings.append(
                f"Classes with < {min_test} samples in test: "
                f"{report.classes_with_few_test}"
            )
    except Exception as e:
        report.warnings.append(f"Split simulation failed: {e}")

    # Suspicious windows: flat or all-zero
    if X.ndim >= 2:
        # X shape: (n_windows, window_len, n_channels) or (n_windows, n_features)
        axis = tuple(range(1, X.ndim))
        var_per_window = np.var(X, axis=axis)
        if var_per_window.ndim > 1:
            var_per_window = np.min(var_per_window, axis=1)
        report.flat_window_count = int((var_per_window < VARIANCE_THRESHOLD_FLAT).sum())
        report.zero_window_count = int(
            (np.abs(X).sum(axis=axis) < 1e-12).sum()
            if np.issubdtype(X.dtype, np.floating)
            else (X.sum(axis=axis) == 0).sum()
        )
        if report.flat_window_count > 0 or report.zero_window_count > 0:
            report.warnings.append(
                f"Suspicious windows: {report.flat_window_count} flat (low variance), "
                f"{report.zero_window_count} all-zero"
            )

    # Verdict: pass if no critical issues
    critical = (
        report.x_nan_count > 0
        or report.x_inf_count > 0
        or report.y_nan_count > 0
        or report.y_inf_count > 0
        or report.num_classes < 2
        or report.total_windows < 20
    )
    report.passed = not critical
    if report.total_windows < 20:
        report.warnings.append("Very few samples (< 20); training may be unreliable.")

    return report
