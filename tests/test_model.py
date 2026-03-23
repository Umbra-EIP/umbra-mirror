"""Test that the EMG → movement model builds correctly."""

from src.emg_movement.model import build_cnn_lstm


def test_build_cnn_lstm():
    model = build_cnn_lstm(input_shape=(200, 10), num_classes=52)
    assert model is not None
    assert model.input_shape == (None, 200, 10)
    assert model.output_shape == (None, 52)
