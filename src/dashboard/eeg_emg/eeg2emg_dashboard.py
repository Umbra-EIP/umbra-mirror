"""EEG→EMG decoder dashboard: load paired .npz + PyTorch checkpoint, run validation inference."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.config import EEG_EMG_DATA_DIR, EEG_EMG_MODEL_DIR
from src.eeg_emg.eeg2emg_inference import run_inference

st.set_page_config(page_title="EEG→EMG Decoder | Umbra", layout="wide", page_icon="🧠")
st.title("🧠 EEG → EMG Decoder")
st.caption(
    "Load a paired EEG/EMG `.npz` and a checkpoint from `eeg2emg_run.py`, then evaluate on a "
    "validation split (same windowing as training)."
)

st.sidebar.header("Data (.npz)")
if not os.path.isdir(EEG_EMG_DATA_DIR):
    os.makedirs(EEG_EMG_DATA_DIR, exist_ok=True)
npz_files = sorted(
    f for f in os.listdir(EEG_EMG_DATA_DIR) if f.endswith(".npz") and not f.startswith(".")
)
uploaded_npz = st.sidebar.file_uploader("Or upload .npz", type=["npz"], accept_multiple_files=False)

st.sidebar.header("Model (.pth)")
if not os.path.isdir(EEG_EMG_MODEL_DIR):
    st.sidebar.error("EEG–EMG model directory missing.")
    pth_files = []
else:
    pth_files = sorted(
        f for f in os.listdir(EEG_EMG_MODEL_DIR) if f.endswith(".pth") and not f.startswith(".")
    )
uploaded_pth = st.sidebar.file_uploader("Or upload .pth", type=["pth"], accept_multiple_files=False)

st.sidebar.header("Options")
step = st.sidebar.number_input("Window step", min_value=1, value=128, step=1)
val_ratio = st.sidebar.slider("Validation fraction", 0.05, 0.5, 0.2, 0.05)
normalize = st.sidebar.checkbox("Z-score normalize (per window)", value=True)
no_cuda = st.sidebar.checkbox("Force CPU", value=False)
resample_eeg = st.sidebar.checkbox("Resample EEG to EMG rate", value=False)
eeg_fs = st.sidebar.number_input("EEG sampling rate (Hz)", min_value=1.0, value=250.0)
emg_fs = st.sidebar.number_input("EMG sampling rate (Hz)", min_value=1.0, value=200.0)

selected_npz: str | None = None
if uploaded_npz is not None:
    tmp_dir = Path(EEG_EMG_DATA_DIR) / "_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / uploaded_npz.name
    dest.write_bytes(uploaded_npz.getvalue())
    selected_npz = str(dest)
elif npz_files:
    selected_npz = os.path.join(EEG_EMG_DATA_DIR, st.sidebar.selectbox("Dataset file", npz_files))

selected_pth: str | None = None
if uploaded_pth is not None:
    tmp_dir = Path(EEG_EMG_MODEL_DIR) / "_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / uploaded_pth.name
    dest.write_bytes(uploaded_pth.getvalue())
    selected_pth = str(dest)
elif pth_files:
    selected_pth = os.path.join(EEG_EMG_MODEL_DIR, st.sidebar.selectbox("Checkpoint", pth_files))

if not npz_files and uploaded_npz is None:
    st.info(
        f"Place paired EEG/EMG `.npz` files under `{EEG_EMG_DATA_DIR}/` or upload one in the sidebar."
    )
if not pth_files and uploaded_pth is None:
    st.info(
        f"Place PyTorch checkpoints (`.pth`) under `{EEG_EMG_MODEL_DIR}/` or upload one in the "
        "sidebar."
    )

run = st.button("Run inference", type="primary")

if run and selected_npz and selected_pth:
    try:
        with st.spinner("Running validation inference…"):
            result = run_inference(
                selected_npz,
                selected_pth,
                step=int(step),
                val_ratio=float(val_ratio),
                normalize=normalize,
                no_cuda=no_cuda,
                resample_eeg=resample_eeg,
                eeg_fs=float(eeg_fs) if resample_eeg else None,
                emg_fs=float(emg_fs) if resample_eeg else None,
            )
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
    else:
        st.subheader("Metrics")
        c1, c2 = st.columns(2)
        c1.metric("Validation MSE", f"{result.mse:.6f}")
        c2.metric("Validation R²", f"{result.r2:.4f}")

        st.subheader("Example: EMG channel 0 (first validation window)")
        preds = result.preds
        trues = result.trues
        if preds.size and trues.size:
            fig, ax = plt.subplots(figsize=(10, 3))
            t_axis = np.arange(preds.shape[-1])
            ax.plot(t_axis, trues[0, 0, :], label="True", color="black", alpha=0.75)
            ax.plot(t_axis, preds[0, 0, :], label="Predicted", color="tab:blue", alpha=0.85)
            ax.set_xlabel("Time (samples)")
            ax.set_ylabel("Amplitude (normalized)")
            ax.legend()
            ax.set_title("First batch item — EMG channel 0")
            st.pyplot(fig)
            plt.close(fig)
