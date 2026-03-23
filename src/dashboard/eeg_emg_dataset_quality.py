"""Quality checks for paired EEG/EMG `.npz` files used in EEG→EMG research."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import EEG_EMG_DATA_DIR, EEG_EMG_QUALITY_REPORTS_DIR
from src.eeg_emg.eeg2emg_inference import prepare_eeg_emg_trials
from src.eeg_emg.eeg2emg_run import EEGEMGWindowDataset, load_npz_with_keys

# Heuristics for research readiness
MIN_TIME_SAMPLES = 64
MIN_TRIALS = 1
VARIANCE_THRESHOLD_FLAT = 1e-12


@dataclass
class EegEmgQualityReport:
    """Quality check result for one `.npz` paired recording."""

    file_path: str
    loaded: bool = False
    error: Optional[str] = None
    report_name: str = "default"

    eeg_key: Optional[str] = None
    emg_key: Optional[str] = None
    eeg_raw_shape: Optional[tuple] = None
    emg_raw_shape: Optional[tuple] = None
    eeg_dtype: Optional[str] = None
    emg_dtype: Optional[str] = None

    eeg_trials_shape: Optional[tuple] = None
    emg_trials_shape: Optional[tuple] = None
    n_trials: int = 0
    n_eeg_channels: int = 0
    n_emg_channels: int = 0
    min_time_samples: int = 0
    time_aligned: bool = True

    eeg_nan: int = 0
    eeg_inf: int = 0
    emg_nan: int = 0
    emg_inf: int = 0

    eeg_min: Optional[float] = None
    eeg_max: Optional[float] = None
    emg_min: Optional[float] = None
    emg_max: Optional[float] = None

    # Sliding-window feasibility (same defaults as training)
    window_size: int = 256
    step: int = 128
    n_windows: int = 0
    pre_windowed: bool = False

    flat_eeg_channels: int = 0
    flat_emg_channels: int = 0
    warnings: list[str] = field(default_factory=list)
    passed: bool = False


def _safe_filename_stem(path: str) -> str:
    base = Path(path).stem
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", base).strip("_")
    return safe or "dataset"


def _report_to_dict(r: EegEmgQualityReport) -> dict[str, Any]:
    d = asdict(r)
    for k in ("eeg_raw_shape", "emg_raw_shape", "eeg_trials_shape", "emg_trials_shape"):
        if d.get(k) is not None:
            d[k] = list(d[k])
    return d


def _report_from_dict(d: dict[str, Any]) -> EegEmgQualityReport:
    d = dict(d)
    for k in ("eeg_raw_shape", "emg_raw_shape", "eeg_trials_shape", "emg_trials_shape"):
        if d.get(k) is not None:
            d[k] = tuple(d[k])
    return EegEmgQualityReport(**d)


def get_reports_dir() -> Path:
    path = Path(EEG_EMG_QUALITY_REPORTS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_saved_reports() -> list[Path]:
    """Saved quality reports, newest first."""
    return sorted(
        get_reports_dir().glob("eeg_emg_q_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def save_report(report: EegEmgQualityReport) -> Path:
    """Save report JSON with timestamp prefix for stable listing."""
    stem = _safe_filename_stem(report.file_path)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in report.report_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = get_reports_dir() / f"eeg_emg_q_{ts}_{stem}_{safe}.json"
    with open(path, "w") as f:
        json.dump(_report_to_dict(report), f, indent=2)
    return path


def load_report(path: Path) -> Optional[EegEmgQualityReport]:
    try:
        with open(path) as f:
            d = json.load(f)
        return _report_from_dict(d)
    except Exception:
        return None


def list_npz_files() -> list[str]:
    """Return sorted `.npz` basenames under `EEG_EMG_DATA_DIR`."""
    if not os.path.isdir(EEG_EMG_DATA_DIR):
        return []
    return sorted(
        f for f in os.listdir(EEG_EMG_DATA_DIR) if f.endswith(".npz") and not f.startswith(".")
    )


def _count_flat_channels(signal_2d: np.ndarray, threshold: float) -> int:
    """signal_2d: (time, channels). Count channels with near-zero variance."""
    if signal_2d.ndim != 2:
        return 0
    var = np.var(signal_2d, axis=0)
    return int((var < threshold).sum())


def check_eeg_emg_npz(
    npz_path: str,
    *,
    window_size: int = 256,
    step: int = 128,
    report_name: str = "default",
) -> EegEmgQualityReport:
    """
    Run integrity and research-readiness checks on a paired EEG/EMG `.npz`.

    Uses the same trial preparation as training/inference (`prepare_eeg_emg_trials`).
    """
    report = EegEmgQualityReport(file_path=npz_path, report_name=report_name)
    report.window_size = window_size
    report.step = step

    if not os.path.isfile(npz_path):
        report.error = f"File not found: {npz_path}"
        return report

    try:
        eeg_key, emg_key, eeg_raw, emg_raw = load_npz_with_keys(npz_path)
        report.eeg_key = eeg_key
        report.emg_key = emg_key
        eeg_raw = np.asarray(eeg_raw)
        emg_raw = np.asarray(emg_raw)
        report.eeg_raw_shape = eeg_raw.shape
        report.emg_raw_shape = emg_raw.shape
        report.eeg_dtype = str(eeg_raw.dtype)
        report.emg_dtype = str(emg_raw.dtype)

        report.eeg_nan = int(np.isnan(eeg_raw).sum())
        report.eeg_inf = int(np.isinf(eeg_raw).sum())
        report.emg_nan = int(np.isnan(emg_raw).sum())
        report.emg_inf = int(np.isinf(emg_raw).sum())

        if report.eeg_nan or report.eeg_inf:
            report.warnings.append(
                f"EEG has invalid values: {report.eeg_nan} NaN, {report.eeg_inf} Inf"
            )
        if report.emg_nan or report.emg_inf:
            report.warnings.append(
                f"EMG has invalid values: {report.emg_nan} NaN, {report.emg_inf} Inf"
            )

        if np.issubdtype(eeg_raw.dtype, np.number):
            report.eeg_min = float(np.nanmin(eeg_raw))
            report.eeg_max = float(np.nanmax(eeg_raw))
        if np.issubdtype(emg_raw.dtype, np.number):
            report.emg_min = float(np.nanmin(emg_raw))
            report.emg_max = float(np.nanmax(emg_raw))

        eeg_trials, emg_trials, pre_w = prepare_eeg_emg_trials(npz_path)
        report.pre_windowed = pre_w
        report.eeg_trials_shape = eeg_trials.shape
        report.emg_trials_shape = emg_trials.shape
        report.n_trials = int(eeg_trials.shape[0])
        report.n_eeg_channels = int(eeg_trials.shape[1])
        report.n_emg_channels = int(emg_trials.shape[1])
        t_eeg = int(eeg_trials.shape[2])
        t_emg = int(emg_trials.shape[2])
        report.min_time_samples = min(t_eeg, t_emg)
        report.time_aligned = t_eeg == t_emg

        if not report.time_aligned:
            report.warnings.append(
                f"Time length mismatch after alignment: EEG={t_eeg}, EMG={t_emg} "
                "(training may trim shorter; verify source alignment)."
            )

        dataset = EEGEMGWindowDataset(
            eeg_trials,
            emg_trials,
            window_size=window_size,
            step=step,
            normalize=True,
            pre_windowed=pre_w,
        )
        report.n_windows = len(dataset)
        if report.n_windows == 0:
            report.warnings.append(
                f"No sliding windows could be built (window_size={window_size}, step={step})."
            )

        # Flat-channel sanity on first trial (full time)
        e0 = eeg_trials[0].T  # (time, ch)
        m0 = emg_trials[0].T
        report.flat_eeg_channels = _count_flat_channels(e0, VARIANCE_THRESHOLD_FLAT)
        report.flat_emg_channels = _count_flat_channels(m0, VARIANCE_THRESHOLD_FLAT)
        if report.flat_eeg_channels:
            report.warnings.append(
                f"EEG has {report.flat_eeg_channels} near-flat channel(s) on trial 0 (var < {VARIANCE_THRESHOLD_FLAT})."
            )
        if report.flat_emg_channels:
            report.warnings.append(
                f"EMG has {report.flat_emg_channels} near-flat channel(s) on trial 0."
            )

        report.loaded = True

        critical = (
            report.eeg_nan > 0
            or report.eeg_inf > 0
            or report.emg_nan > 0
            or report.emg_inf > 0
            or report.n_windows == 0
            or report.min_time_samples < MIN_TIME_SAMPLES
            or report.n_trials < MIN_TRIALS
        )
        report.passed = not critical
        if report.min_time_samples < MIN_TIME_SAMPLES:
            report.warnings.append(
                f"Very short recording ({report.min_time_samples} samples); "
                f"consider at least {MIN_TIME_SAMPLES} for stable windows."
            )

    except Exception as exc:
        report.error = str(exc)
        report.warnings.append(f"Exception during check: {exc}")

    return report
