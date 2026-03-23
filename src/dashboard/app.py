"""Streamlit entry: grouped navigation for EMG hand movement vs EEG–EMG."""

from pathlib import Path

import streamlit as st

_dashboard = Path(__file__).resolve().parent

pages = {
    "EMG — Hand movement": [
        st.Page(
            str(_dashboard / "emg" / "hand_movement_decoder.py"),
            title="Decoder",
            icon="🖐️",
            url_path="emg-decoder",
        ),
        st.Page(
            str(_dashboard / "emg" / "dataset_quality.py"),
            title="Dataset quality",
            icon="📋",
            url_path="emg-dataset-quality",
        ),
        st.Page(
            str(_dashboard / "emg" / "model_comparator_page.py"),
            title="Model comparator",
            icon="🔬",
            url_path="emg-model-comparator",
        ),
        st.Page(
            str(_dashboard / "emg" / "hardware_impact_page.py"),
            title="Hardware impact",
            icon="⚡",
            url_path="emg-hardware-impact",
        ),
    ],
    "EEG–EMG": [
        st.Page(
            str(_dashboard / "eeg_emg" / "eeg2emg_dashboard.py"),
            title="Decoder",
            icon="🧠",
            url_path="eeg-emg-decoder",
        ),
        st.Page(
            str(_dashboard / "eeg_emg" / "dataset_quality_page.py"),
            title="Dataset quality",
            icon="📋",
            url_path="eeg-emg-dataset-quality",
        ),
        st.Page(
            str(_dashboard / "eeg_emg" / "model_comparator_page.py"),
            title="Model comparator",
            icon="🔬",
            url_path="eeg-emg-model-comparator",
        ),
        st.Page(
            str(_dashboard / "eeg_emg" / "hardware_impact_page.py"),
            title="Hardware impact",
            icon="⚡",
            url_path="eeg-emg-hardware-impact",
        ),
    ],
}

pg = st.navigation(pages)
pg.run()
