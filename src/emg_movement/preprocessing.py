# src/emg_movement/preprocessing.py

import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from src.config import NINAPRO_ROOT, PREPROCESS_PATH, SAMPLING_EMG_RATE
from src.emg_movement.utils import parse_exercise_data


class EMGPreprocessor:
    def __init__(self, window_size=200, window_step=50):
        self.window_size = window_size
        self.window_step = window_step
        self.exercise_ids = [1, 2, 3]

        self.sos = signal.butter(
            N=2, Wn=5, fs=SAMPLING_EMG_RATE, btype="low", output="sos"
        )

    def preprocess(self):
        subjects = self._load_subjects()
        subject_ids = subjects["Subject"].unique().tolist()

        X_all = []
        y_all = []

        for subject_id in tqdm(subject_ids, desc="Subjects"):
            for ex in self.exercise_ids:
                emg, labels = self._load_exercise(subject_id, ex)

                if emg is None:
                    continue

                X, y = self._windowize(emg, labels)

                X_all.append(X)
                y_all.append(y)

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        print("✅ Final dataset")
        print("X:", X_all.shape)
        print("y:", y_all.shape)

        existing = [
            int(d)
            for d in os.listdir(PREPROCESS_PATH)
            if os.path.isdir(os.path.join(PREPROCESS_PATH, d)) and d.isdigit()
        ]

        next_id = max(existing, default=0) + 1
        save_dir = os.path.join(PREPROCESS_PATH, str(next_id))
        os.makedirs(save_dir, exist_ok=True)

        x_out = os.path.join(save_dir, "X.npy")
        y_out = os.path.join(save_dir, "y.npy")

        try:
            np.save(x_out, X_all)
            np.save(y_out, y_all)
        except Exception:
            raise

    def _load_subjects(self):
        return pd.read_csv(os.path.join(NINAPRO_ROOT, "subjects.csv"))

    def _load_exercise(self, subject_id, exercise_id):
        try:
            data = parse_exercise_data(NINAPRO_ROOT, [subject_id], exercise_id)
        except Exception:
            return None, None

        emg = data[:, :10]

        labels = data[:, 10].astype(int)

        emg = signal.sosfiltfilt(self.sos, emg, axis=0)

        return emg, labels

    def _windowize(self, emg, labels):
        X = []
        y = []

        for start in range(0, len(emg) - self.window_size, self.window_step):
            end = start + self.window_size

            window_emg = emg[start:end]
            window_labels = labels[start:end]

            # Ignore rest
            label = np.bincount(window_labels).argmax()
            if label == 0:
                continue

            X.append(window_emg)
            y.append(label)

        return np.array(X), np.array(y)
