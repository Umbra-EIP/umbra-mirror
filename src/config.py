# src/config.py

FS = 1000

LOWPASS_CUTOFF = 5
FILTER_ORDER = 2

NINAPRO_ROOT = "data/ninapro"
PREPROCESS_PATH = "data/preprocessed"
MODEL_DIR = "src/models"

# EEG→EMG dashboard & data (paired .npz, PyTorch checkpoints)
EEG_EMG_DATA_DIR = "data/eeg_emg"
EEG_EMG_MODEL_DIR = "src/eeg_emg"
# Canonical files auto-selected by all EEG–EMG dashboard pages
EEG_EMG_DEFAULT_NPZ = "data/eeg_emg/dataset_augmented.npz"
EEG_EMG_DEFAULT_PTH = "src/eeg_emg/eeg2emg_best.pth"
# Personalized / single-subject EEG→EMG checkpoint (see eeg2emg_train_single_subject.py)
EEG_EMG_SUBJECT_MODEL_PTH = "src/eeg_emg/eeg2emg_single_subject_best.pth"
EEG_EMG_QUALITY_REPORTS_DIR = "data/eeg_emg_quality_reports"
EEG_EMG_COMPARISON_REPORTS_DIR = "data/eeg_emg_comparison_reports"
EEG_EMG_HARDWARE_REPORTS_DIR = "data/eeg_emg_hardware_reports"

QUALITY_REPORTS_DIR = "data/quality_reports"
COMPARISON_REPORTS_DIR = "data/comparison_reports"
HARDWARE_REPORTS_DIR = "data/hardware_reports"

WINDOW_MS = 500
WINDOW_SAMPLES = int(FS * WINDOW_MS / 1000)
WINDOW_STEP_MS = 100
WINDOW_STEP_SAMPLES = int(FS * WINDOW_STEP_MS / 1000)

REST_LABEL = 0

SAMPLING_EMG_RATE = 100
