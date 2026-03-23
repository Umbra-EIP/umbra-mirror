"""Import-safety tests for EMG training entrypoint."""

import importlib


def test_emg_movement_train_imports_without_side_effects() -> None:
    """Importing the module must not parse sys.argv or load training data."""
    importlib.import_module("src.emg_movement.train")
