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

## Current layout

- Entry: `streamlit run src/dashboard/app.py`
- **EMG — Hand movement**
  - Decoder (`emg/hand_movement_decoder.py`): NinaPro preprocessed `X.npy`/`y.npy`, Keras `.keras` models
  - Dataset quality (`emg/dataset_quality.py`)
  - Model comparator (`emg/model_comparator_page.py`)
  - Hardware impact (`emg/hardware_impact_page.py`)
- **EEG–EMG** (research loop on collected `.npz` + `.pth`)
  - Decoder (`eeg_emg/eeg2emg_dashboard.py`)
  - Dataset quality (`eeg_emg/dataset_quality_page.py`) — `src/dashboard/eeg_emg_dataset_quality.py`, reports `EEG_EMG_QUALITY_REPORTS_DIR`
  - Model comparator (`eeg_emg/model_comparator_page.py`) — `src/dashboard/eeg_emg_model_compare.py`, `EEG_EMG_COMPARISON_REPORTS_DIR`
  - Hardware impact (`eeg_emg/hardware_impact_page.py`) — `src/dashboard/eeg_emg_torch_hardware.py`, `EEG_EMG_HARDWARE_REPORTS_DIR`
  - Inference / windowing: `src/eeg_emg/eeg2emg_inference.py`; NPZ keys: `load_npz_with_keys()` in `eeg2emg_run.py`
  - Paths: `EEG_EMG_DATA_DIR`, `EEG_EMG_MODEL_DIR` + report dirs in `src/config.py`

Issues [#72](https://github.com/Umbra-EIP/umbra-mirror/issues/72)–[#74](https://github.com/Umbra-EIP/umbra-mirror/issues/74) are covered by this stack.

## Follow-up ideas

- Quality page: per-channel histograms or PSD preview.
- Comparator: overlay predicted vs true EMG for two checkpoints on the same window.
- Hardware: optional CUDA peak memory during a longer stress loop.

## Verification

- `ruff check .`
- `pytest`
- Manual: start Streamlit, confirm both sections and all EMG pages behave as before; EEG–EMG decoder runs with a sample `.npz` + matching `.pth`.
