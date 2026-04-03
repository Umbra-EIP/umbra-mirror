"""Hardware profiling for EEG→EMG PyTorch checkpoints (load + inference latency)."""

from __future__ import annotations

import gc
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import EEG_EMG_HARDWARE_REPORTS_DIR, EEG_EMG_MODEL_DIR
from src.eeg_emg.eeg2emg_inference import load_eeg2emg_model, prepare_eeg_emg_trials
from src.eeg_emg.eeg2emg_run import EEGEMGWindowDataset


def _process_ram_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _torch_cuda_mem_mb() -> Optional[float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        return None


@dataclass
class TorchSystemInfo:
    platform_str: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    device_name: str


@dataclass
class TorchLoadProfile:
    model_file: str
    file_size_mb: float
    load_time_s: float
    ram_before_mb: float
    ram_after_mb: float
    cuda_mem_after_mb: Optional[float]
    n_eeg_channels: Optional[int] = None
    n_emg_channels: Optional[int] = None
    window_size: Optional[int] = None
    n_params_total: Optional[int] = None
    n_params_trainable: Optional[int] = None
    load_error: Optional[str] = None


@dataclass
class TorchInferenceLatency:
    batch_size: int
    n_batches: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float


@dataclass
class EegEmgTorchHardwareReport:
    generated_at: str
    npz_path: str
    model_file: str
    system: dict[str, Any]
    loading: dict[str, Any]
    inference: list[dict[str, Any]]
    config: dict[str, Any] = field(default_factory=dict)


def get_torch_system_info() -> TorchSystemInfo:
    """Collect Python / PyTorch / CUDA environment."""
    import torch

    cuda_ver = torch.version.cuda if torch.cuda.is_available() else None
    dev = "cpu"
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)

    return TorchSystemInfo(
        platform_str=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=cuda_ver,
        device_name=dev,
    )


def get_dashboard_system_context() -> dict[str, Any]:
    """
    Merge PyTorch/CUDA info with host RAM and CPU counts for the hardware dashboard.

    Used so the EEG–EMG hardware page can mirror the EMG tracker system section.
    """
    ti = get_torch_system_info()
    out: dict[str, Any] = asdict(ti)
    try:
        import psutil

        mem = psutil.virtual_memory()
        out["ram_total_mb"] = mem.total / (1024 * 1024)
        out["ram_available_mb"] = mem.available / (1024 * 1024)
        out["cpu_count_physical"] = psutil.cpu_count(logical=False)
        out["cpu_count_logical"] = psutil.cpu_count(logical=True) or 1
    except Exception:
        out["ram_total_mb"] = 0.0
        out["ram_available_mb"] = 0.0
        out["cpu_count_physical"] = None
        out["cpu_count_logical"] = os.cpu_count() or 1
    return out


def run_torch_profile(
    npz_path: str,
    model_file: str,
    *,
    model_dir: str = EEG_EMG_MODEL_DIR,
    batch_sizes: Optional[list[int]] = None,
    n_batch_runs: int = 5,
    step: int = 128,
    val_ratio: float = 0.2,
    normalize: bool = True,
    seed: int = 42,
    no_cuda: bool = False,
) -> EegEmgTorchHardwareReport:
    """
    Load a checkpoint and measure forward-pass latency on the validation subset.

    Uses the same windowing and split as ``run_inference``.
    """
    import torch
    from torch.utils.data import DataLoader, random_split

    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16]

    path = os.path.join(model_dir, model_file)
    loading = TorchLoadProfile(
        model_file=model_file,
        file_size_mb=os.path.getsize(path) / (1024 * 1024) if os.path.isfile(path) else 0.0,
        load_time_s=0.0,
        ram_before_mb=_process_ram_mb(),
        ram_after_mb=0.0,
        cuda_mem_after_mb=None,
    )

    device = torch.device("cpu" if no_cuda else ("cuda" if torch.cuda.is_available() else "cpu"))

    gc.collect()
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    cfg: dict[str, Any] = {}
    try:
        model, cfg = load_eeg2emg_model(path, map_location=device)
        loading.n_eeg_channels = cfg.get("n_eeg_channels")
        loading.n_emg_channels = cfg.get("n_emg_channels")
        loading.window_size = cfg.get("window_size")
        loading.n_params_total = sum(p.numel() for p in model.parameters())
        loading.n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception as exc:
        loading.load_error = str(exc)
        loading.load_time_s = time.perf_counter() - t0
        return EegEmgTorchHardwareReport(
            generated_at=datetime.now().isoformat(timespec="seconds"),
            npz_path=npz_path,
            model_file=model_file,
            system=get_dashboard_system_context(),
            loading=asdict(loading),
            inference=[],
            config={"batch_sizes": batch_sizes, "error": "load_failed"},
        )

    loading.load_time_s = time.perf_counter() - t0
    loading.ram_after_mb = _process_ram_mb()
    loading.cuda_mem_after_mb = _torch_cuda_mem_mb()

    ws = int(cfg.get("window_size", 256))
    eeg_trials, emg_trials, pre_w = prepare_eeg_emg_trials(npz_path)
    dataset = EEGEMGWindowDataset(
        eeg_trials,
        emg_trials,
        window_size=ws,
        step=step,
        normalize=normalize,
        pre_windowed=pre_w,
    )

    n_total = len(dataset)
    if n_total == 0:
        return EegEmgTorchHardwareReport(
            generated_at=datetime.now().isoformat(timespec="seconds"),
            npz_path=npz_path,
            model_file=model_file,
            system=get_dashboard_system_context(),
            loading=asdict(loading),
            inference=[],
            config={"error": "no_windows"},
        )

    if n_total == 1:
        val_set = dataset
    else:
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        if n_train < 1:
            n_train, n_val = n_total - 1, 1
        gen = torch.Generator().manual_seed(seed)
        _, val_set = random_split(dataset, [n_train, n_val], generator=gen)

    inference_rows: list[dict[str, Any]] = []
    model.eval()

    for bs in batch_sizes:
        loader = DataLoader(val_set, batch_size=bs, shuffle=False)
        times: list[float] = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_batch_runs:
                    break
                xb, _ = batch
                xb = xb.to(device).float()
                if xb.size(0) < 1:
                    continue
                t1 = time.perf_counter()
                _ = model(xb)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t1) * 1000.0)

        if not times:
            continue
        arr = np.array(times)
        inference_rows.append(
            asdict(
                TorchInferenceLatency(
                    batch_size=bs,
                    n_batches=len(times),
                    mean_ms=float(np.mean(arr)),
                    std_ms=float(np.std(arr)),
                    p50_ms=float(np.percentile(arr, 50)),
                    p95_ms=float(np.percentile(arr, 95)),
                )
            )
        )

    return EegEmgTorchHardwareReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        npz_path=npz_path,
        model_file=model_file,
        system=get_dashboard_system_context(),
        loading=asdict(loading),
        inference=inference_rows,
        config={
            "batch_sizes": batch_sizes,
            "n_batch_runs": n_batch_runs,
            "step": step,
            "val_ratio": val_ratio,
            "device": str(device),
            "window_size": ws,
        },
    )


def get_hardware_reports_dir() -> Path:
    p = Path(EEG_EMG_HARDWARE_REPORTS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_torch_hardware_report(report: EegEmgTorchHardwareReport, name: str = "default") -> Path:
    """Persist a torch hardware report."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = get_hardware_reports_dir() / f"eeg_emg_hw_{ts}_{safe}.json"
    payload = {"name": name, "saved_at": report.generated_at, "report": asdict(report)}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def list_saved_torch_hardware_reports() -> list[tuple[str, str, Path]]:
    d = get_hardware_reports_dir()
    if not d.is_dir():
        return []
    out: list[tuple[str, str, Path]] = []
    for p in sorted(d.glob("eeg_emg_hw_*.json"), reverse=True):
        try:
            with open(p) as f:
                meta = json.load(f)
            out.append((meta.get("name", p.stem), meta.get("saved_at", ""), p))
        except Exception:
            continue
    return out


def load_torch_hardware_payload(path: Path) -> Optional[dict[str, Any]]:
    """Load full JSON payload (name, saved_at, report)."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
