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
        ),
        st.Page(
            str(_dashboard / "emg" / "dataset_quality.py"),
            title="Dataset quality",
            icon="📋",
        ),
        st.Page(
            str(_dashboard / "emg" / "model_comparator_page.py"),
            title="Model comparator",
            icon="🔬",
        ),
        st.Page(
            str(_dashboard / "emg" / "hardware_impact_page.py"),
            title="Hardware impact",
            icon="⚡",
        ),
    ],
    "EEG–EMG": [
        st.Page(
            str(_dashboard / "eeg_emg" / "eeg2emg_dashboard.py"),
            title="Decoder",
            icon="🧠",
        ),
    ],
}

pg = st.navigation(pages)
pg.run()
