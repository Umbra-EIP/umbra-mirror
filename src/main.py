# Preprocess EMG NinaPro data and save to data/preprocessed/<id>/X.npy, y.npy.
# Run from repo root: python -m src.main

from src.emg_movement.preprocessing import EMGPreprocessor

if __name__ == "__main__":
    pre = EMGPreprocessor()
    pre.preprocess()
