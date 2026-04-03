"""EEG→EMG decoder dashboard: auto-loads canonical checkpoint, shows rich per-channel metrics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.eeg_emg.dashboard_defaults import (
    EEG_EMG_DATA_DIR,
    EEG_EMG_DEFAULT_NPZ,
    EEG_EMG_DEFAULT_PTH,
    EEG_EMG_MODEL_DIR,
    EEG_EMG_SUBJECT_MODEL_PTH,
)
from src.eeg_emg.eeg2emg_inference import run_inference

st.set_page_config(page_title="EEG→EMG Decoder | Umbra", layout="wide")
st.title("EEG → EMG Decoder")
st.caption("Validation inference with per-channel metrics and interactive waveform viewer.")

# ── Sidebar — file selection ──────────────────────────────────────────────────
st.sidebar.header("Data (.npz)")
os.makedirs(EEG_EMG_DATA_DIR, exist_ok=True)
npz_files = sorted(
    f for f in os.listdir(EEG_EMG_DATA_DIR) if f.endswith(".npz") and not f.startswith(".")
)
uploaded_npz = st.sidebar.file_uploader("Override: upload .npz", type=["npz"])

default_npz_name = Path(EEG_EMG_DEFAULT_NPZ).name
npz_index = npz_files.index(default_npz_name) if default_npz_name in npz_files else 0
if uploaded_npz is not None:
    tmp = Path(EEG_EMG_DATA_DIR) / "_uploads" / uploaded_npz.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(uploaded_npz.getvalue())
    selected_npz = str(tmp)
elif npz_files:
    choice = st.sidebar.selectbox("Dataset file", npz_files, index=npz_index)
    selected_npz = os.path.join(EEG_EMG_DATA_DIR, choice)
else:
    selected_npz = None
    st.info(f"Place `.npz` files under `{EEG_EMG_DATA_DIR}/`.")

st.sidebar.header("Model (.pth)")
pth_files: list[str] = []
if os.path.isdir(EEG_EMG_MODEL_DIR):
    pth_files = sorted(
        f for f in os.listdir(EEG_EMG_MODEL_DIR) if f.endswith(".pth") and not f.startswith(".")
    )
uploaded_pth = st.sidebar.file_uploader("Override: upload .pth", type=["pth"])

general_name = Path(EEG_EMG_DEFAULT_PTH).name
subject_name = Path(EEG_EMG_SUBJECT_MODEL_PTH).name
preset = st.sidebar.radio(
    "Checkpoint preset",
    ["General", "Single-subject"],
    horizontal=True,
    help="Single-subject: checkpoint from eeg2emg_train_single_subject.py (higher val accuracy on one subject).",
)
if preset == "Single-subject" and subject_name in pth_files:
    pth_index = pth_files.index(subject_name)
elif general_name in pth_files:
    pth_index = pth_files.index(general_name)
else:
    pth_index = 0
if uploaded_pth is not None:
    tmp = Path(EEG_EMG_MODEL_DIR) / "_uploads" / uploaded_pth.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(uploaded_pth.getvalue())
    selected_pth = str(tmp)
elif pth_files:
    choice_pth = st.sidebar.selectbox("Checkpoint", pth_files, index=pth_index)
    selected_pth = os.path.join(EEG_EMG_MODEL_DIR, choice_pth)
else:
    selected_pth = None
    st.info(f"Place `.pth` checkpoints under `{EEG_EMG_MODEL_DIR}/`.")

st.sidebar.header("Options")
step = st.sidebar.number_input("Window step", min_value=1, value=128, step=1)
val_ratio = st.sidebar.slider("Validation fraction", 0.05, 0.5, 0.2, 0.05)
normalize = st.sidebar.checkbox("Z-score normalize", value=True)
no_cuda = st.sidebar.checkbox("Force CPU", value=False)
resample_eeg = st.sidebar.checkbox("Resample EEG to EMG rate", value=False)
eeg_fs = st.sidebar.number_input("EEG fs (Hz)", min_value=1.0, value=250.0)
emg_fs = st.sidebar.number_input("EMG fs (Hz)", min_value=1.0, value=200.0)

# ── Run / cache ───────────────────────────────────────────────────────────────
run = st.button("Run inference", type="primary", disabled=not (selected_npz and selected_pth))

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
        st.session_state["eeg_emg_result"] = result
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.stop()

result = st.session_state.get("eeg_emg_result")
if result is None:
    st.info("Click **Run inference** to evaluate the selected checkpoint on the validation split.")
    st.stop()

m = result.metrics
preds = result.preds  # (N, C, T)
trues = result.trues

# ── Model config banner ───────────────────────────────────────────────────────
if result.model_cfg:
    cfg = result.model_cfg
    st.caption(
        f"**Model:** EEG ch={cfg.get('n_eeg_channels')}  EMG ch={cfg.get('n_emg_channels')}  "
        f"CNN={cfg.get('cnn_channels')}  LSTM={cfg.get('lstm_hidden')}×{cfg.get('lstm_layers')}  "
        f"bidir={cfg.get('bidirectional')}  window={cfg.get('window_size')} samples  "
        f"Val windows={preds.shape[0]}"
    )

# ── Global metrics ────────────────────────────────────────────────────────────
st.subheader("Global metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("MSE", f"{m.mse:.5f}")
c2.metric("RMSE", f"{m.rmse:.5f}")
c3.metric("MAE", f"{m.mae:.5f}")
c4.metric("R²", f"{m.r2:.4f}")

c5, c6 = st.columns(2)
c5.metric("Mean Pearson r", f"{m.mean_pearson:.4f}")
c6.metric("Val windows", preds.shape[0])

# ── Per-channel metrics table ─────────────────────────────────────────────────
st.subheader("Per-channel metrics")
n_ch = m.n_channels
ch_rows = []
for ch in range(n_ch):
    ch_rows.append(
        {
            "Channel": f"EMG {ch}",
            "Pearson r": round(m.pearson_per_channel[ch], 4),
            "R²": round(m.r2_per_channel[ch], 4),
            "MAE": round(m.mae_per_channel[ch], 5),
            "Envelope corr": round(m.envelope_corr_per_channel[ch], 4),
        }
    )
df_ch = pd.DataFrame(ch_rows)


def _colour(val: float) -> str:
    g = int(min(max(val, 0), 1) * 180)
    return f"background-color: rgba(0,{g},80,0.15)"


st.dataframe(
    df_ch.style.applymap(_colour, subset=["Pearson r", "R²", "Envelope corr"]).format(
        {"Pearson r": "{:.4f}", "R²": "{:.4f}", "MAE": "{:.5f}", "Envelope corr": "{:.4f}"}
    ),
    use_container_width=True,
    hide_index=True,
)

# ── Interactive waveform viewer ───────────────────────────────────────────────
st.subheader("Waveform viewer")
col_ch, col_win = st.columns(2)
with col_ch:
    sel_ch = st.selectbox("EMG channel", list(range(n_ch)), format_func=lambda c: f"CH {c}")
with col_win:
    sel_win = st.slider("Validation window", 0, max(preds.shape[0] - 1, 0), 0)

p_sig = preds[sel_win, sel_ch, :]
t_sig = trues[sel_win, sel_ch, :]
t_axis = np.arange(len(p_sig))
residual = p_sig - t_sig

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
ax1.plot(t_axis, t_sig, label="True", color="black", alpha=0.8, linewidth=1.2)
ax1.plot(t_axis, p_sig, label="Predicted", color="#2196F3", alpha=0.85, linewidth=1.0)
ax1.fill_between(t_axis, t_sig, p_sig, alpha=0.15, color="#2196F3", label="Error band")
ax1.set_ylabel("Amplitude (norm.)")
ax1.legend(loc="upper right", fontsize=8)
ax1.set_title(f"EMG CH {sel_ch}  —  window {sel_win}")
ax1.grid(alpha=0.3)

ax2.plot(t_axis, residual, color="#E53935", linewidth=0.9, label="Residual (pred − true)")
ax2.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax2.set_xlabel("Time (samples)")
ax2.set_ylabel("Residual")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

window_pearson = m.pearson_per_channel[sel_ch]
st.caption(
    f"Channel {sel_ch} overall Pearson r = **{window_pearson:.4f}** · "
    f"R² = **{m.r2_per_channel[sel_ch]:.4f}** · "
    f"Envelope corr = **{m.envelope_corr_per_channel[sel_ch]:.4f}**"
)
