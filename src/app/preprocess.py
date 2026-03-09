import numpy as np
from scipy import signal
import streamlit as st


class EMGPreprocessor:
    def __init__(self, window_size=200, window_step=50, fs=2000):
        self.window_size = window_size
        self.window_step = window_step

        self.sos = signal.butter(N=2, Wn=5, fs=fs, btype="low", output="sos")

    def preprocess(self, emg, labels):
        """
        emg: (T, C)
        labels: (T,)
        """
        emg = signal.sosfiltfilt(self.sos, emg, axis=0)

        X, y = [], []
        total_steps = (len(emg) - self.window_size) // self.window_step
        progress = st.progress(0.0)

        step = 0
        for start in range(0, len(emg) - self.window_size, self.window_step):
            end = start + self.window_size

            window_emg = emg[start:end]
            window_labels = labels[start:end]

            label = np.bincount(window_labels).argmax()
            if label != 0:
                X.append(window_emg)
                y.append(label)

            step += 1
            progress.progress(step / total_steps)

        return np.array(X), np.array(y)
