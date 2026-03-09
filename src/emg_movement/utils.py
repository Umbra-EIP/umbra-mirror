import numpy as np
import pandas as pd

import zipfile
import os
from tqdm import tqdm
import scipy.io as sio


def unzip_and_remove(zip_folder: str) -> None:
    zip_files = [f for f in os.listdir(zip_folder) if f.endswith(".zip")]

    for zip_file in tqdm(zip_files):
        zip_path = os.path.join(zip_folder, zip_file)
        unzip_path = os.path.join(zip_folder, zip_file.replace(".zip", ""))

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        os.remove(zip_path)


def parse_exercise_data(
    datapath: str, subject_ids: list, exercise_id: int
) -> np.ndarray:
    all_data = []
    for subject_id in tqdm(
        subject_ids, desc=f"Loading data for exercise {exercise_id}"
    ):
        filepath = os.path.join(
            datapath, f"s{subject_id}", f"S{subject_id}_A1_E{exercise_id}.mat"
        )
        data = sio.loadmat(filepath)

        emg, stimulus, repetition, subject_id_arr = (
            data["emg"],
            data["restimulus"],
            data["rerepetition"],
            np.repeat(subject_id, data["emg"].shape[0]).reshape(-1, 1),
        )
        data = np.concatenate([emg, stimulus, repetition, subject_id_arr], axis=1)

        all_data.append(data)

    all_data = np.concatenate(all_data, axis=0)

    return all_data


def train_val_test_split(df: pd.DataFrame, val: list, test: list) -> pd.DataFrame:
    df["Split"] = "train"

    df.loc[df["Repetition"].isin(val), "Split"] = "val"

    df.loc[df["Repetition"].isin(test), "Split"] = "test"

    df.loc[df["Repetition"] == 0, "Split"] = None

    df["Split"] = df["Split"].ffill()

    df = df.dropna()

    return df


def extract_trial_windows(
    emg: np.ndarray, df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    labels = df["Stimulus"].values
    repetitions = df["Repetition"].values

    curr_stim, curr_rep = None, None

    window_ids = []

    curr_id = 0
    for y, rep in zip(labels, repetitions):
        if curr_stim is None:
            curr_rep, curr_stim = rep, y

        if (y == curr_stim) and (rep == curr_rep):
            window_ids.append(curr_id)

        else:
            curr_stim, curr_rep = y, rep

            curr_id += 1

            window_ids.append(curr_id)

    df.loc[:, "Window"] = window_ids

    min_length = df.groupby("Window").size().min()

    n_windows = len(df["Window"].unique())
    feat_dim = emg.shape[1]

    result_array = np.empty((n_windows, min_length, feat_dim))
    labels = np.zeros((n_windows,), dtype=int)

    for i, window_value in enumerate(df["Window"].unique()):
        window_indices = df[df["Window"] == window_value].index[:min_length]

        result_array[i, :, :] = emg[window_indices, :]

        stimulus = df[df["Window"] == window_value]["Stimulus"].values[0]
        labels[i] = stimulus

    return result_array, labels


def extract_time_windows(
    emg: np.ndarray,
    labels: np.ndarray,
    sampling_frequency: int,
    win_len: int,
    step: int,
) -> tuple[np.array, np.array]:
    n, m = emg.shape

    win_len = int(win_len * sampling_frequency)

    start_points = np.arange(0, n - win_len, int(step * sampling_frequency))
    end_points = start_points + win_len

    emg_windows = np.zeros((len(start_points), win_len, m))
    labels_windows = []

    for i in range(len(start_points)):
        # Extract the EMG data
        emg_windows[i, :, :] = emg[start_points[i] : end_points[i], :]

        # Extract the labels
        labels_window = labels[start_points[i] : end_points[i]]

        # Get the most frequent label
        val, count = np.unique(labels_window, return_counts=True)
        most_frequent_label = val[np.argmax(count)]

        labels_windows.append(most_frequent_label)

    return emg_windows, np.array(labels_windows, dtype=int)


def calc_fft_power(
    emg_windows: np.array, sampling_frequency: int
) -> tuple[np.array, np.array]:
    N = emg_windows.shape[1]

    freqs = np.fft.rfftfreq(N, 1 / sampling_frequency)

    # Fast Fourier Transform (FFT)
    fft_vals = np.fft.rfft(emg_windows, axis=1)
    fft_power = np.abs(fft_vals) ** 2  # Power spectrum

    return freqs[1:], fft_power[:, 1:, :]


def downsample_rest_windows(
    data: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    X, y = data

    _, count = np.unique(y, return_counts=True)

    avg_count_stim = int(np.mean(count[1:]))

    rest_count = count[0]
    n_samples_to_remove = rest_count - avg_count_stim

    rest_index = np.where(y == 0)[0]

    remove_index = np.random.choice(rest_index, size=n_samples_to_remove, replace=False)

    X = np.delete(X, remove_index, axis=0)
    y = np.delete(y, remove_index, axis=0)

    return X, y


def extract_features(emg_windows: np.ndarray, sampling_frequency: int) -> np.ndarray:
    mav = np.mean(np.abs(emg_windows), axis=1)

    maxav = np.max(np.abs(emg_windows), axis=1)

    std = np.std(emg_windows, axis=1)

    rms = np.sqrt(np.mean(emg_windows**2, axis=1))

    wl = np.sum(np.abs(np.diff(emg_windows, axis=1)), axis=1)

    freqs, fft_power = calc_fft_power(
        emg_windows, sampling_frequency=sampling_frequency
    )

    mean_power = np.mean(fft_power, axis=1)

    tot_power = np.sum(fft_power, axis=1)

    freqs_reshaped = freqs.reshape(1, freqs.shape[0], 1)
    mean_frequency = np.sum(fft_power * freqs_reshaped, axis=1) / tot_power

    cumulative_power = np.cumsum(fft_power, axis=1)
    total_power = cumulative_power[:, -1, :]
    median_frequency = np.zeros((emg_windows.shape[0], emg_windows.shape[2]))

    for i in range(emg_windows.shape[0]):
        for j in range(emg_windows.shape[2]):
            median_frequency[i, j] = freqs[
                np.where(cumulative_power[i, :, j] >= total_power[i, j] / 2)[0][0]
            ]

    peak_frequency = freqs[np.argmax(fft_power, axis=1)]

    X = np.column_stack(
        (
            mav,
            maxav,
            std,
            rms,
            wl,
            mean_power,
            tot_power,
            mean_frequency,
            median_frequency,
            peak_frequency,
        )
    )

    return X


def drop_missing_values(data: tuple[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X, y = data

    missing_index = np.where(np.isnan(X))[0]

    X = np.delete(X, missing_index, axis=0)
    y = np.delete(y, missing_index, axis=0)

    return X, y


def collect_subject_data(
    subject_ids: list[int], subjects_features: dict
) -> tuple[np.ndarray, np.ndarray]:
    X_all, y_all = None, None
    for subject in subject_ids:
        X_train, y_train, X_val, y_val, X_test, y_test = subjects_features[subject]

        X = np.concatenate([X_train, X_val, X_test], axis=0)
        y = np.concatenate([y_train, y_val, y_test], axis=0)

        if X_all is None:
            X_all, y_all = X, y
        else:
            X_all = np.concatenate([X_all, X], axis=0)
            y_all = np.concatenate([y_all, y], axis=0)

    return X_all, y_all
