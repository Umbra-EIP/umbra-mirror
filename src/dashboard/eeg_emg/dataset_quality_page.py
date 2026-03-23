# EEG–EMG dataset quality — Streamlit page

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st

from src.config import EEG_EMG_DATA_DIR
from src.dashboard.eeg_emg_dataset_quality import (
    check_eeg_emg_npz,
    list_npz_files,
    list_saved_reports,
    load_report,
    save_report,
)

st.set_page_config(
    page_title="EEG–EMG Dataset Quality | Umbra",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📋 EEG–EMG Dataset Quality")
st.caption(
    "Validate paired `.npz` archives (EEG + EMG keys, shapes, NaNs, sliding-window count) before "
    "training or benchmarking."
)

st.sidebar.header("Dataset (.npz)")
if not os.path.isdir(EEG_EMG_DATA_DIR):
    os.makedirs(EEG_EMG_DATA_DIR, exist_ok=True)
npz_files = list_npz_files()
uploaded = st.sidebar.file_uploader("Or upload .npz", type=["npz"], accept_multiple_files=False)

selected_path: str | None = None
if uploaded is not None:
    dest_dir = Path(EEG_EMG_DATA_DIR) / "_uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / uploaded.name
    dest.write_bytes(uploaded.getvalue())
    selected_path = str(dest)
elif npz_files:
    choice = st.sidebar.selectbox("File", npz_files)
    selected_path = os.path.join(EEG_EMG_DATA_DIR, choice)

window_size = st.sidebar.number_input("Window size (for window count)", 32, 4096, 256, 32)
step = st.sidebar.number_input("Step", 1, 2048, 128, 1)
report_name = st.sidebar.text_input("Report name", value="default").strip() or "default"

if st.sidebar.button("Run quality check", type="primary") and selected_path:
    with st.spinner("Checking…"):
        rep = check_eeg_emg_npz(
            selected_path, window_size=int(window_size), step=int(step), report_name=report_name
        )
    st.session_state["eeg_emg_q_report"] = rep
    try:
        p = save_report(rep)
        st.sidebar.success(f"Saved `{p.name}`")
    except Exception as exc:
        st.sidebar.warning(f"Save failed: {exc}")

st.sidebar.divider()
st.sidebar.subheader("Saved reports")
saved = list_saved_reports()
if saved:
    chosen_path = st.sidebar.selectbox("Load report", options=saved, format_func=lambda p: p.name)
    if st.sidebar.button("Load"):
        loaded = load_report(chosen_path)
        if loaded:
            st.session_state["eeg_emg_q_report"] = loaded
        else:
            st.sidebar.error("Failed to load.")

if not npz_files and uploaded is None:
    st.info(f"Add `.npz` files under `{EEG_EMG_DATA_DIR}/` or upload one.")
    st.stop()

rep = st.session_state.get("eeg_emg_q_report")
if rep is None:
    st.info("Run a quality check from the sidebar.")
    st.stop()

verdict = "PASS" if rep.passed else "FAIL"
st.markdown(f"### Verdict: **{verdict}**")
if rep.error:
    st.error(rep.error)

cols = st.columns(4)
cols[0].metric("EEG trials shape", str(rep.eeg_trials_shape or "—"))
cols[1].metric("EMG trials shape", str(rep.emg_trials_shape or "—"))
cols[2].metric("Sliding windows", rep.n_windows)
cols[3].metric("Min time (samples)", rep.min_time_samples)

detail = {
    "eeg_key": rep.eeg_key,
    "emg_key": rep.emg_key,
    "pre_windowed": rep.pre_windowed,
    "time_aligned": rep.time_aligned,
    "eeg_nan/inf": f"{rep.eeg_nan} / {rep.eeg_inf}",
    "emg_nan/inf": f"{rep.emg_nan} / {rep.emg_inf}",
    "flat EEG channels (trial 0)": rep.flat_eeg_channels,
    "flat EMG channels (trial 0)": rep.flat_emg_channels,
}
st.dataframe(pd.DataFrame([detail]), use_container_width=True)

if rep.warnings:
    st.subheader("Warnings")
    for w in rep.warnings:
        st.warning(w)
