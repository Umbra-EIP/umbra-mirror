# src/emg_movement/model.py

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_lstm(input_shape=(200, 10), num_classes=52):
    """Build the EMG → movement CNN-LSTM model."""
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(32, 5, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 5, activation="relu"),
            layers.Conv1D(128, 3, activation="relu"),
            layers.MaxPooling1D(2),
            layers.LSTM(64, return_sequences=False),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model
