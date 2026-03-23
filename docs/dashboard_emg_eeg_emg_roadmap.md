# Dashboard: EMG hand movement vs EEG–EMG

This document tracks the split of the Streamlit app into two product areas and parity work for **EEG→EMG** with **EMG→hand movement**.

## GitHub issues

| # | Title |
|---|--------|
| [70](https://github.com/Umbra-EIP/umbra-mirror/issues/70) | Split EMG vs EEG–EMG navigation and relocate EMG pages |
| [71](https://github.com/Umbra-EIP/umbra-mirror/issues/71) | EEG–EMG decoder (npz + checkpoint, metrics, visualization) |
| [72](https://github.com/Umbra-EIP/umbra-mirror/issues/72) | EEG–EMG dataset quality checker for `.npz` pairs |
| [73](https://github.com/Umbra-EIP/umbra-mirror/issues/73) | EEG–EMG model comparator (PyTorch regression) |
| [74](https://github.com/Umbra-EIP/umbra-mirror/issues/74) | EEG–EMG hardware impact tracker (PyTorch) |

## Current layout (after #70 / #71)

- Entry: `streamlit run src/dashboard/app.py`
- **EMG — Hand movement**
  - Decoder (`emg/hand_movement_decoder.py`): NinaPro preprocessed `X.npy`/`y.npy`, Keras `.keras` models
  - Dataset quality (`emg/dataset_quality.py`)
  - Model comparator (`emg/model_comparator_page.py`)
  - Hardware impact (`emg/hardware_impact_page.py`)
- **EEG–EMG**
  - Decoder (`eeg_emg/eeg2emg_dashboard.py`): paired `.npz`, PyTorch `.pth` from `eeg2emg_run.py`
  - Inference helpers: `src/eeg_emg/eeg2emg_inference.py`
  - Paths: `EEG_EMG_DATA_DIR`, `EEG_EMG_MODEL_DIR` in `src/config.py`

## Remaining steps (by issue)

### #72 — Dataset quality (EEG–EMG)

1. Define quality rules for paired EEG/EMG arrays (shape alignment, NaNs, channel counts, duration).
2. Add `src/dashboard/eeg_emg/dataset_quality_page.py` (or similar) and optional report directory (e.g. `data/eeg_emg_quality_reports/`).
3. Reuse UX patterns from `src/dashboard/dataset_quality.py` where sensible.

### #73 — Model comparator (EEG–EMG)

1. List/compare `.pth` checkpoints (metadata from saved `config`, file mtime, size).
2. Run a fixed validation protocol (same `run_inference` or shared evaluator) and persist JSON reports.
3. Optional: overlay prediction curves for two models.

### #74 — Hardware impact (EEG–EMG)

1. Port `hardware_profiler` patterns: Torch model load time, RAM, GPU memory (if CUDA), batch latency.
2. New page under EEG–EMG section; save reports under e.g. `data/eeg_emg_hardware_reports/` (add to `src/config.py` when introduced).

## Verification

- `ruff check .`
- `pytest`
- Manual: start Streamlit, confirm both sections and all EMG pages behave as before; EEG–EMG decoder runs with a sample `.npz` + matching `.pth`.
