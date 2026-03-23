"""Backend utilities for the Hardware Impact Tracker dashboard page."""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.config import HARDWARE_REPORTS_DIR, MODEL_DIR, PREPROCESS_PATH

ProgressFn = Callable[[str, int, int], None]

_DEFAULT_INPUT_SHAPE = (200, 10)  # fallback for this project's CNN-LSTM models


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class GpuInfo:
    name: str
    total_memory_mb: float


@dataclass
class SystemInfo:
    platform_str: str
    python_version: str
    cpu_count_physical: Optional[int]
    cpu_count_logical: int
    ram_total_mb: float
    ram_available_mb: float
    gpu_devices: list  # list[GpuInfo as dict]
    tensorflow_version: str
    has_gpu: bool


@dataclass
class LoadingProfile:
    model_name: str
    file_size_mb: float
    file_modified: Optional[str]
    param_count: Optional[int]
    model_input_shape: Optional[list]  # stored as list for JSON compat
    load_time_s: float
    ram_before_mb: float
    ram_after_mb: float
    ram_delta_mb: float
    gpu_mem_before_mb: Optional[float]
    gpu_mem_after_mb: Optional[float]
    gpu_mem_delta_mb: Optional[float]
    loaded: bool
    load_error: Optional[str]


@dataclass
class BatchInferenceResult:
    batch_size: int
    n_runs: int
    mean_latency_ms: float
    std_latency_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    per_sample_ms: float  # mean_latency_ms / batch_size
    throughput_samples_per_sec: float  # batch_size / (mean_ms / 1000)


@dataclass
class StressTestResult:
    n_inferences: int
    batch_size: int
    total_time_s: float
    mean_latency_ms: float
    std_latency_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    peak_ram_mb: float
    peak_gpu_mb: Optional[float]
    throughput_samples_per_sec: float
    latency_timeline: list  # list[float], one per batch inference call


@dataclass
class HardwareReport:
    generated_at: str
    model_name: str
    system_info: dict
    loading_profile: Optional[dict]
    batch_inference: list  # list[dict from BatchInferenceResult]
    stress_test: Optional[dict]
    dataset_id: Optional[str]
    dataset_shape: Optional[list]  # stored as list for JSON compat
    config: dict


# ── Public helpers ────────────────────────────────────────────────────────────


def get_available_models() -> list[str]:
    """Return sorted list of .keras filenames in MODEL_DIR."""
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".keras"))


def get_available_datasets() -> list[str]:
    """Return sorted list of preprocessed dataset IDs."""
    if not os.path.isdir(PREPROCESS_PATH):
        return []
    return sorted(
        d
        for d in os.listdir(PREPROCESS_PATH)
        if os.path.isdir(os.path.join(PREPROCESS_PATH, d))
        and os.path.exists(os.path.join(PREPROCESS_PATH, d, "X.npy"))
    )


# ── Memory helpers ────────────────────────────────────────────────────────────


def _get_process_ram_mb() -> float:
    """Return current process RSS in MB; 0.0 if psutil is unavailable."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _get_system_ram() -> tuple[float, float]:
    """Return (total_mb, available_mb); zeros if psutil is unavailable."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.total / (1024 * 1024), mem.available / (1024 * 1024)
    except Exception:
        return 0.0, 0.0


def _get_cpu_counts() -> tuple[Optional[int], int]:
    """Return (physical_cores, logical_cores)."""
    try:
        import psutil

        return psutil.cpu_count(logical=False), psutil.cpu_count(logical=True) or 1
    except Exception:
        return None, os.cpu_count() or 1


def _get_gpu_memory_mb() -> Optional[float]:
    """Return currently allocated TF GPU memory in MB, or None if not available."""
    try:
        import tensorflow as tf

        if not tf.config.list_physical_devices("GPU"):
            return None
        info = tf.config.experimental.get_memory_info("GPU:0")
        return info.get("current", 0) / (1024 * 1024)
    except Exception:
        return None


# ── System info ───────────────────────────────────────────────────────────────


def get_system_info() -> SystemInfo:
    """Gather platform, CPU, RAM, and GPU information."""
    import tensorflow as tf

    total_mb, avail_mb = _get_system_ram()
    phys, logical = _get_cpu_counts()

    gpu_devices: list[dict] = []
    has_gpu = False
    try:
        for dev in tf.config.list_physical_devices("GPU"):
            has_gpu = True
            details: dict[str, Any] = tf.config.experimental.get_device_details(dev)
            gpu_devices.append(
                {
                    "name": details.get("device_name", dev.name),
                    "total_memory_mb": details.get("memory_limit", 0) / (1024 * 1024),
                }
            )
    except Exception:
        pass

    return SystemInfo(
        platform_str=platform.platform(),
        python_version=platform.python_version(),
        cpu_count_physical=phys,
        cpu_count_logical=logical,
        ram_total_mb=total_mb,
        ram_available_mb=avail_mb,
        gpu_devices=gpu_devices,
        tensorflow_version=tf.__version__,
        has_gpu=has_gpu,
    )


# ── File helpers ──────────────────────────────────────────────────────────────


def _file_modified_iso(name: str) -> Optional[str]:
    try:
        ts = os.path.getmtime(os.path.join(MODEL_DIR, name))
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return None


def _file_size_mb(name: str) -> float:
    try:
        return os.path.getsize(os.path.join(MODEL_DIR, name)) / (1024 * 1024)
    except OSError:
        return 0.0


# ── Dataset loader ────────────────────────────────────────────────────────────


def _load_dataset_sample(
    dataset_id: str, n_windows: int, seed: int = 42
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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


def _make_synthetic_data(input_shape: tuple, n_samples: int) -> np.ndarray:
    """Generate random float32 data for profiling when no real dataset is available."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_samples, *input_shape)).astype(np.float32)


# ── Core profiling functions ──────────────────────────────────────────────────


def profile_model_loading(model_name: str) -> tuple[LoadingProfile, Optional[Any]]:
    """Load the model while measuring RAM/GPU memory and wall-clock time.

    Returns (LoadingProfile, model_or_None).  The caller is responsible for
    reusing the returned model object to avoid loading it a second time.
    """
    import tensorflow as tf

    gc.collect()
    ram_before = _get_process_ram_mb()
    gpu_before = _get_gpu_memory_mb()

    path = os.path.join(MODEL_DIR, model_name)
    size_mb = _file_size_mb(model_name)
    modified = _file_modified_iso(model_name)

    t0 = time.perf_counter()
    try:
        model = tf.keras.models.load_model(path)
        t1 = time.perf_counter()

        param_count = int(model.count_params())
        try:
            input_shape = [int(d) if d is not None else -1 for d in model.input_shape[1:]]
        except Exception:
            input_shape = None

        ram_after = _get_process_ram_mb()
        gpu_after = _get_gpu_memory_mb()
        gpu_delta = (
            (gpu_after - gpu_before) if (gpu_after is not None and gpu_before is not None) else None
        )

        return (
            LoadingProfile(
                model_name=model_name,
                file_size_mb=size_mb,
                file_modified=modified,
                param_count=param_count,
                model_input_shape=input_shape,
                load_time_s=t1 - t0,
                ram_before_mb=ram_before,
                ram_after_mb=ram_after,
                ram_delta_mb=ram_after - ram_before,
                gpu_mem_before_mb=gpu_before,
                gpu_mem_after_mb=gpu_after,
                gpu_mem_delta_mb=gpu_delta,
                loaded=True,
                load_error=None,
            ),
            model,
        )
    except Exception as exc:
        t1 = time.perf_counter()
        ram_after = _get_process_ram_mb()
        return (
            LoadingProfile(
                model_name=model_name,
                file_size_mb=size_mb,
                file_modified=modified,
                param_count=None,
                model_input_shape=None,
                load_time_s=t1 - t0,
                ram_before_mb=ram_before,
                ram_after_mb=ram_after,
                ram_delta_mb=ram_after - ram_before,
                gpu_mem_before_mb=gpu_before,
                gpu_mem_after_mb=_get_gpu_memory_mb(),
                gpu_mem_delta_mb=None,
                loaded=False,
                load_error=str(exc),
            ),
            None,
        )


def profile_batch_inference(
    model: Any,
    X: np.ndarray,
    batch_sizes: list[int],
    n_runs: int = 5,
    progress_callback: Optional[ProgressFn] = None,
) -> list[BatchInferenceResult]:
    """Measure mean/std/percentile inference latency across multiple batch sizes.

    Each batch size is timed `n_runs` times (after a warm-up pass).
    The batch is taken from the first `batch_size` rows of X; if X has fewer
    rows, the whole array is used.
    """
    results: list[BatchInferenceResult] = []
    total = len(batch_sizes)

    # Warm-up with a single sample to initialise TF graph
    model.predict(X[:1], verbose=0)

    for i, bs in enumerate(batch_sizes):
        if progress_callback:
            progress_callback(f"Batch size {bs} — timing…", i, total)

        batch = X[: min(bs, len(X))]
        times_ms: list[float] = []

        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.predict(batch, verbose=0, batch_size=bs)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        arr = np.array(times_ms)
        mean_ms = float(np.mean(arr))
        actual_bs = len(batch)
        throughput = (actual_bs * n_runs) / (float(np.sum(arr)) / 1000.0)

        results.append(
            BatchInferenceResult(
                batch_size=actual_bs,
                n_runs=n_runs,
                mean_latency_ms=mean_ms,
                std_latency_ms=float(np.std(arr)),
                p50_ms=float(np.percentile(arr, 50)),
                p95_ms=float(np.percentile(arr, 95)),
                p99_ms=float(np.percentile(arr, 99)),
                per_sample_ms=mean_ms / actual_bs,
                throughput_samples_per_sec=throughput,
            )
        )

    return results


def run_stress_test(
    model: Any,
    X: np.ndarray,
    n_inferences: int = 1000,
    batch_size: int = 1,
    progress_callback: Optional[ProgressFn] = None,
) -> StressTestResult:
    """Run `n_inferences` batched predict() calls; track latency and peak memory.

    Each call uses `batch_size` samples cycled from X.  RAM and GPU memory are
    sampled every ~5 % of iterations to estimate peak usage.
    """
    try:
        import psutil

        proc: Optional[Any] = psutil.Process(os.getpid())
    except Exception:
        proc = None

    # Warm-up
    model.predict(X[:batch_size], verbose=0)

    times_ms: list[float] = []
    peak_ram = _get_process_ram_mb()
    peak_gpu = _get_gpu_memory_mb()
    sample_every = max(n_inferences // 20, 1)

    t_total_start = time.perf_counter()

    for i in range(n_inferences):
        start_idx = (i * batch_size) % max(len(X) - batch_size + 1, 1)
        batch = X[start_idx : start_idx + batch_size]
        if len(batch) < batch_size:
            batch = X[:batch_size]

        t0 = time.perf_counter()
        model.predict(batch, verbose=0, batch_size=batch_size)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

        if i % sample_every == 0:
            if proc is not None:
                try:
                    cur_ram = proc.memory_info().rss / (1024 * 1024)
                    peak_ram = max(peak_ram, cur_ram)
                except Exception:
                    pass
            cur_gpu = _get_gpu_memory_mb()
            if cur_gpu is not None:
                peak_gpu = max(peak_gpu or 0.0, cur_gpu)

            if progress_callback:
                progress_callback(
                    f"Stress test: {i + 1}/{n_inferences} inferences…",
                    i,
                    n_inferences,
                )

    t_total = time.perf_counter() - t_total_start
    arr = np.array(times_ms)

    return StressTestResult(
        n_inferences=n_inferences,
        batch_size=batch_size,
        total_time_s=t_total,
        mean_latency_ms=float(np.mean(arr)),
        std_latency_ms=float(np.std(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        peak_ram_mb=peak_ram,
        peak_gpu_mb=peak_gpu,
        throughput_samples_per_sec=(n_inferences * batch_size) / t_total,
        latency_timeline=times_ms,
    )


# ── Main orchestrator ─────────────────────────────────────────────────────────


def run_hardware_profile(
    model_name: str,
    dataset_id: Optional[str] = None,
    n_dataset_windows: int = 500,
    batch_sizes: Optional[list[int]] = None,
    n_batch_runs: int = 5,
    run_stress: bool = False,
    n_stress_inferences: int = 500,
    stress_batch_size: int = 1,
    progress_callback: Optional[ProgressFn] = None,
) -> HardwareReport:
    """End-to-end hardware profiling pipeline.

    Steps:
      1. Gather system info.
      2. Load model with memory/time profiling.
      3. Prepare data (real dataset or synthetic).
      4. Profile batch inference latency across multiple batch sizes.
      5. (optional) Run stress test.
      6. Return a serialisable HardwareReport.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32, 64]

    def _cb(label: str, done: int, total: int) -> None:
        if progress_callback:
            progress_callback(label, done, total)

    _cb("Gathering system information…", 0, 10)
    sys_info = get_system_info()

    _cb(f"Loading model {model_name}…", 1, 10)
    loading_profile, model = profile_model_loading(model_name)

    # ── Prepare data ──────────────────────────────────────────────────────────
    X: Optional[np.ndarray] = None
    dataset_shape: Optional[list] = None

    if dataset_id is not None:
        _cb(f"Loading dataset {dataset_id}…", 2, 10)
        X, _ = _load_dataset_sample(dataset_id, n_dataset_windows)
        if X is not None:
            dataset_shape = list(X.shape[1:])

    if X is None and model is not None:
        # Fall back to synthetic data so we can still profile inference
        if loading_profile.model_input_shape is not None:
            in_shape = tuple(
                d if d != -1 else _DEFAULT_INPUT_SHAPE[i]
                for i, d in enumerate(loading_profile.model_input_shape)
            )
        else:
            in_shape = _DEFAULT_INPUT_SHAPE

        n_synth = max(max(batch_sizes), 200)
        _cb(f"Generating synthetic data (shape {in_shape}, n={n_synth})…", 2, 10)
        X = _make_synthetic_data(in_shape, n_synth)

    # ── Batch inference profiling ─────────────────────────────────────────────
    batch_results: list[BatchInferenceResult] = []
    if model is not None and X is not None:
        _cb("Profiling batch inference…", 3, 10)
        batch_results = profile_batch_inference(
            model, X, batch_sizes, n_runs=n_batch_runs, progress_callback=_cb
        )

    # ── Stress test ───────────────────────────────────────────────────────────
    stress_result: Optional[StressTestResult] = None
    if run_stress and model is not None and X is not None:
        _cb("Starting stress test…", 7, 10)
        stress_result = run_stress_test(
            model,
            X,
            n_inferences=n_stress_inferences,
            batch_size=stress_batch_size,
            progress_callback=_cb,
        )

    _cb("Assembling report…", 9, 10)
    report = HardwareReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        model_name=model_name,
        system_info=asdict(sys_info),
        loading_profile=asdict(loading_profile),
        batch_inference=[asdict(b) for b in batch_results],
        stress_test=asdict(stress_result) if stress_result is not None else None,
        dataset_id=dataset_id,
        dataset_shape=dataset_shape,
        config={
            "n_dataset_windows": n_dataset_windows,
            "batch_sizes": batch_sizes,
            "n_batch_runs": n_batch_runs,
            "run_stress": run_stress,
            "n_stress_inferences": n_stress_inferences,
            "stress_batch_size": stress_batch_size,
            "used_synthetic_data": X is not None and dataset_id is None,
        },
    )
    _cb("Done", 10, 10)
    return report


# ── Persistence ───────────────────────────────────────────────────────────────


def _json_default(obj: Any) -> Any:
    """Fallback JSON serialiser for types not handled by default."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_report(report: HardwareReport, name: str = "default") -> Path:
    """Persist a HardwareReport to disk as JSON."""
    os.makedirs(HARDWARE_REPORTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(HARDWARE_REPORTS_DIR) / f"hardware_{ts}_{name}.json"
    payload = {
        "name": name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "report": asdict(report),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


def load_report(path: Path) -> HardwareReport:
    """Load a HardwareReport from a JSON file."""
    with open(path) as f:
        payload = json.load(f)
    d = payload["report"]
    return HardwareReport(
        generated_at=d.get("generated_at", ""),
        model_name=d.get("model_name", ""),
        system_info=d.get("system_info", {}),
        loading_profile=d.get("loading_profile"),
        batch_inference=d.get("batch_inference", []),
        stress_test=d.get("stress_test"),
        dataset_id=d.get("dataset_id"),
        dataset_shape=d.get("dataset_shape"),
        config=d.get("config", {}),
    )


def list_saved_reports() -> list[tuple[str, str, Path]]:
    """Return (name, saved_at, path) tuples sorted newest-first."""
    dir_path = Path(HARDWARE_REPORTS_DIR)
    if not dir_path.is_dir():
        return []
    items: list[tuple[str, str, Path]] = []
    for p in sorted(dir_path.glob("hardware_*.json"), reverse=True):
        try:
            with open(p) as f:
                meta = json.load(f)
            items.append((meta.get("name", p.stem), meta.get("saved_at", ""), p))
        except Exception:
            continue
    return items
