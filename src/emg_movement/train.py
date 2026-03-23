# src/emg_movement/train.py

import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import MODEL_DIR, PREPROCESS_PATH
from src.emg_movement.model import build_cnn_lstm

parser = argparse.ArgumentParser(description="Train EMG → movement CNN-LSTM")
parser.add_argument(
    "--dataset",
    type=int,
    default=1,
    help="Preprocessed dataset id (e.g. 1 for data/preprocessed/1/)",
)
parser.add_argument(
    "--output",
    type=str,
    default="cnn_lstm_emg_v3.keras",
    help="Output model filename under src/models/",
)


def main() -> None:
    """Load preprocessed EMG data, train CNN-LSTM, and save the Keras model."""
    args = parser.parse_args()

    data_dir = os.path.join(PREPROCESS_PATH, str(args.dataset))
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    y = y - 1

    print(X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_classes = len(np.unique(y))

    model = build_cnn_lstm(input_shape=(200, 10), num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, args.output))


if __name__ == "__main__":
    main()
