import os

import numpy as np
import scipy.io as sio
from scipy.signal import resample

# ============================================================
# config
# ============================================================

EEG_DIR = r"Dataset\EEG_DATA\data\mat_data\subject_1"
EMG_DIR = r"Dataset\EMG_DATA\data\mat_data\subject_1"

OUTPUT_NPZ = "dataset_subject_1.npz"

EEG_FS = 250
EMG_FS = 200


# ============================================================
# 1) get and append .mat files
# ============================================================


def list_pairs(eeg_dir, emg_dir):
    eeg_files = sorted([f for f in os.listdir(eeg_dir) if f.endswith(".mat")])
    emg_files = sorted([f for f in os.listdir(emg_dir) if f.endswith(".mat")])

    pairs = []

    for eeg_file in eeg_files:
        base = eeg_file.replace(".mat", "")
        if base in [e.replace(".mat", "") for e in emg_files]:
            pairs.append((os.path.join(eeg_dir, eeg_file), os.path.join(emg_dir, eeg_file)))

    print(f"{len(pairs)} paires trouvées")
    return pairs


# ============================================================
# 2) load the EEG/EMG file
# ============================================================


def load_mat_pair(eeg_path, emg_path):
    eeg = sio.loadmat(eeg_path)["data"]  # (8, N)
    emg = sio.loadmat(emg_path)["data"]  # (M, 8)

    eeg = eeg.T  # → (N, 8)

    return eeg, emg


# ============================================================
# 3) resampling EEG (250 Hz → 200 Hz)
# ============================================================


def resample_eeg(eeg):
    orig_samples = eeg.shape[0]
    new_samples = int(orig_samples * (EMG_FS / EEG_FS))
    eeg_resampled = resample(eeg, new_samples, axis=0)
    return eeg_resampled


# ============================================================
# 4) lenghts alignement
# ============================================================


def align(eeg, emg):
    min_len = min(len(eeg), len(emg))
    return eeg[:min_len], emg[:min_len]


# ============================================================
# 5) MAIN PROCESS
# ============================================================


def main():
    pairs = list_pairs(EEG_DIR, EMG_DIR)

    all_eeg = []
    all_emg = []

    for eeg_path, emg_path in pairs:
        print("\nProcessing pair:")
        print("EEG:", eeg_path)
        print("EMG:", emg_path)

        eeg_raw, emg_raw = load_mat_pair(eeg_path, emg_path)

        eeg_resampled = resample_eeg(eeg_raw)
        eeg_sync, emg_sync = align(eeg_resampled, emg_raw)

        print("  -> aligned shape:", eeg_sync.shape)

        all_eeg.append(eeg_sync)
        all_emg.append(emg_sync)

    EEG_FULL = np.concatenate(all_eeg, axis=0)
    EMG_FULL = np.concatenate(all_emg, axis=0)

    np.savez(OUTPUT_NPZ, EEG=EEG_FULL, EMG=EMG_FULL)

    print("\n================================================")
    print("Dataset :", OUTPUT_NPZ)
    print("EEG final :", EEG_FULL.shape)
    print("EMG final :", EMG_FULL.shape)
    print("================================================")


if __name__ == "__main__":
    main()
