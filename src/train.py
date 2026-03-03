# src/train.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

X = np.load("data/preprocessed/X.npy")
y = np.load("data/preprocessed/y.npy")
y = y - 1


print(X.shape, y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_classes = len(np.unique(y))

model = models.Sequential(
    [
        layers.Input(shape=(200, 10)),
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


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
)

model.save("src/models/cnn_lstm_emg_v3.keras")
