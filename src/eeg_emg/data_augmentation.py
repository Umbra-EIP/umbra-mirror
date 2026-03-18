import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# Fake data generator
# ==========================================
def create_dummy_dataset(filename="dataset_brut.npz"):
    print(">>> Génération d'un faux dataset pour l'exemple...")
    n_samples = 250 * 60
    eeg_data = np.sin(np.linspace(0, 100, n_samples)).reshape(-1, 1) + np.random.normal(
        0, 0.5, (n_samples, 8)
    )
    emg_data = np.zeros((n_samples, 2))
    for i in range(0, n_samples, 500):
        if i + 100 < n_samples:
            emg_data[i : i + 100, :] = np.random.normal(0, 2, (100, 2))  # Burst EMG

    np.savez(filename, eeg=eeg_data, emg=emg_data)
    print(f"    Fichier '{filename}' créé avec succès.\n")


# ==========================================
# Data augmentation
# ==========================================


def add_gaussian_noise(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def scale_amplitude(data, min_scale=0.8, max_scale=1.2):
    factor = np.random.uniform(min_scale, max_scale)
    return data * factor


def create_sliding_windows(X, y, window_size, overlap_ratio):
    step_size = int(window_size * (1 - overlap_ratio))
    X_windows = []
    y_windows = []

    for start_idx in range(0, len(X) - window_size, step_size):
        end_idx = start_idx + window_size

        X_slice = X[start_idx:end_idx]
        y_slice = y[start_idx:end_idx]

        X_windows.append(X_slice)
        y_windows.append(y_slice)

    return np.array(X_windows), np.array(y_windows)


# ==========================================
# main pipeline
# ==========================================


def process_dataset(input_file):
    print(f">>> Loading {input_file}...")
    data = np.load(input_file)

    try:
        raw_eeg = data["EEG"]  # Input (Brain)
        raw_emg = data["EMG"]  # Target (Muscle)
    except KeyError:
        print("Error: 'eeg' or 'emg' keys are not found in the .npz")
        print("Available keys :", data.files)
        return

    print(f"    base shape EEG: {raw_eeg.shape} (Temps x Canaux)")
    print(f"    base shape EMG: {raw_emg.shape}")

    SAMPLING_RATE = 250
    WINDOW_SECONDS = 2
    OVERLAP = 0.90

    window_samples = int(WINDOW_SECONDS * SAMPLING_RATE)

    print(f">>> Sliding Window (Overlap {OVERLAP * 100}%)...")
    X_aug, y_aug = create_sliding_windows(raw_eeg, raw_emg, window_samples, OVERLAP)

    print(f"    New augmented dataset : {X_aug.shape[0]} exemples")

    print(">>> applying Noise Injection and Scaling...")

    X_final = []
    y_final = []

    for i in range(len(X_aug)):
        X_final.append(X_aug[i])
        y_final.append(y_aug[i])

        X_noisy = add_gaussian_noise(X_aug[i], noise_level=0.1)
        X_final.append(X_noisy)
        y_final.append(y_aug[i])  # Final EMG signal stays the same

        X_scaled = scale_amplitude(X_aug[i])
        # IMPORTANT: if EEG is stronger, the EMG doesnt follow directly the trend,
        # we only scale input X for robustness.
        X_final.append(X_scaled)
        y_final.append(y_aug[i])

    X_final = np.array(X_final)
    y_final = np.array(y_final)

    print(f"    Final dataset size after augmentation: {X_final.shape}")

    output_filename = "dataset_augmented.npz"
    np.savez(output_filename, X=X_final, y=y_final)
    print(f">>> Sauvegardé sous '{output_filename}'")

    plot_check(X_final, y_final, 0)


def plot_check(X, y, index):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(X[index])
    plt.title(f"Exemple {index} - EEG (Inputs)")
    plt.ylabel("Amplitude (uV)")

    plt.subplot(2, 1, 2)
    plt.plot(y[index])
    plt.title(f"Exemple {index} - EMG (Targets)")
    plt.ylabel("Amplitude (uV)")
    plt.xlabel("Time steps")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # create_dummy_dataset() # comment or uncomment to use false data or not

    process_dataset("dataset_subject_1.npz")
