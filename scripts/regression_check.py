#!/usr/bin/env python3
"""Model accuracy regression gate.

Loads a trained Keras model and evaluates it on the held-out test split that
mirrors train.py (test_size=0.2, random_state=42, stratified by class label).
Exits non-zero when accuracy falls below the configured threshold.

Safe to run in CI where data / model may not be present:
  - missing model file → prints [SKIP], exits 0
  - missing data files → prints [SKIP], exits 0

Usage:
    python scripts/regression_check.py [--model PATH] [--data-dir DIR] [--threshold FLOAT]

Examples:
    python scripts/regression_check.py
    python scripts/regression_check.py --model src/models/cnn_lstm_emg_v2.keras --threshold 0.75
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regression check — ensures model accuracy stays above threshold"
    )
    p.add_argument(
        "--model",
        default="src/models/cnn_lstm_emg_v3.keras",
        help="Path to the .keras model file (default: src/models/cnn_lstm_emg_v3.keras)",
    )
    p.add_argument(
        "--data-dir",
        default="data/preprocessed/1",
        help="Directory containing X.npy and y.npy (default: data/preprocessed/1)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Minimum acceptable test accuracy in [0, 1] (default: 0.80)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    data_dir = Path(args.data_dir)

    if not model_path.exists():
        print(f"[SKIP] Model not found: {model_path}", file=sys.stderr)
        return 0

    if not (data_dir / "X.npy").exists() or not (data_dir / "y.npy").exists():
        print(f"[SKIP] Preprocessed data not found at {data_dir}", file=sys.stderr)
        return 0

    # Heavy imports deferred so the script exits fast on skip paths above
    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    print(f"Loading model : {model_path}")
    model: tf.keras.Model = tf.keras.models.load_model(model_path)

    print(f"Loading data  : {data_dir}")
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    y = y - 1  # match train.py: convert to 0-indexed labels

    # Reproduce the exact held-out split used during training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Test samples  : {len(X_test)}")
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    passed = accuracy >= args.threshold
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] accuracy={accuracy:.4f}  threshold={args.threshold:.4f}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
