# src/preprocessing.py

import os
import json
import traceback
import time
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from config import *
import scripts.utils as utils

_DEBUG_LOG_PATH = "/Users/merlish/codage/umbra/.cursor/debug-054783.log"
_DEBUG_SESSION_ID = "054783"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict):
    # #region agent log
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion agent log


def _disk_free_bytes(path: str) -> int:
    try:
        st = os.statvfs(path)
        return int(st.f_bavail * st.f_frsize)
    except Exception:
        return -1


class EMGPreprocessor:
    def __init__(self, window_size=200, window_step=50):
        self.window_size = window_size
        self.window_step = window_step
        self.exercise_ids = [1, 2, 3]

        self.sos = signal.butter(
            N=2, Wn=5, fs=SAMPLING_EMG_RATE, btype="low", output="sos"
        )

    def preprocess(self):
        _debug_log(
            "H_ENTRY",
            "src/preprocessing.py:preprocess",
            "preprocess_start",
            {
                "window_size": self.window_size,
                "window_step": self.window_step,
                "preprocess_path": PREPROCESS_PATH,
            },
        )
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

        _debug_log(
            "H_SHAPES",
            "src/preprocessing.py:preprocess",
            "final_dataset_ready",
            {
                "X_shape": list(X_all.shape),
                "y_shape": list(y_all.shape),
                "X_dtype": str(X_all.dtype),
                "y_dtype": str(y_all.dtype),
                "X_nbytes": int(getattr(X_all, "nbytes", -1)),
                "y_nbytes": int(getattr(y_all, "nbytes", -1)),
            },
        )

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
        free_bytes = _disk_free_bytes(save_dir)

        _debug_log(
            "H_SAVE_ENV",
            "src/preprocessing.py:preprocess",
            "about_to_save",
            {
                "save_dir": save_dir,
                "x_out": x_out,
                "y_out": y_out,
                "existing_ids": existing[:50],
                "next_id": next_id,
                "cwd": os.getcwd(),
                "free_bytes_at_save_dir": free_bytes,
            },
        )

        try:
            np.save(x_out, X_all)
            _debug_log(
                "H_SAVE",
                "src/preprocessing.py:preprocess",
                "saved_X_ok",
                {"x_out": x_out},
            )
            np.save(y_out, y_all)
            _debug_log(
                "H_SAVE",
                "src/preprocessing.py:preprocess",
                "saved_y_ok",
                {"y_out": y_out},
            )
        except Exception as e:
            _debug_log(
                "H_SAVE_FAIL",
                "src/preprocessing.py:preprocess",
                "save_failed",
                {
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(limit=8),
                    "x_out_exists": os.path.exists(x_out),
                    "y_out_exists": os.path.exists(y_out),
                    "free_bytes_at_save_dir": _disk_free_bytes(save_dir),
                },
            )
            raise

    def _load_subjects(self):
        return pd.read_csv(os.path.join(NINAPRO_ROOT, "subjects.csv"))

    def _load_exercise(self, subject_id, exercise_id):
        try:
            data = utils.parse_exercise_data(NINAPRO_ROOT, [subject_id], exercise_id)
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
