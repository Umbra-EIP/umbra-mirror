# EEG–EMG dataset quality — Streamlit page
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

from src.dashboard.eeg_emg_dataset_quality import (
    check_eeg_emg_npz,
    list_npz_files,
    list_saved_reports,
    load_report,
    save_report,
)
from src.eeg_emg.dashboard_defaults import EEG_EMG_DATA_DIR, EEG_EMG_DEFAULT_NPZ
from src.eeg_emg.eeg2emg_inference import prepare_eeg_emg_trials

st.set_page_config(
    page_title="EEG–EMG Dataset Quality | Umbra",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("EEG–EMG Dataset Quality")
st.caption(
    "Validate paired `.npz` archives, inspect signal statistics, channel correlations, "
    "and power spectra."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Dataset (.npz)")
os.makedirs(EEG_EMG_DATA_DIR, exist_ok=True)
npz_files = list_npz_files()
uploaded = st.sidebar.file_uploader("Override: upload .npz", type=["npz"])

default_name = Path(EEG_EMG_DEFAULT_NPZ).name
npz_index = npz_files.index(default_name) if default_name in npz_files else 0

selected_path: str | None = None
if uploaded is not None:
    dest_dir = Path(EEG_EMG_DATA_DIR) / "_uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / uploaded.name
    dest.write_bytes(uploaded.getvalue())
    selected_path = str(dest)
elif npz_files:
    choice = st.sidebar.selectbox("File", npz_files, index=npz_index)
    selected_path = os.path.join(EEG_EMG_DATA_DIR, choice)

window_size = st.sidebar.number_input("Window size", 32, 4096, 256, 32)
step = st.sidebar.number_input("Step", 1, 2048, 128, 1)
eeg_fs = st.sidebar.number_input("EEG fs (Hz)", 1.0, 10000.0, 250.0)
report_name = st.sidebar.text_input("Report name", value="default").strip() or "default"

if st.sidebar.button("Run quality check", type="primary") and selected_path:
    with st.spinner("Checking…"):
        rep = check_eeg_emg_npz(
            selected_path, window_size=int(window_size), step=int(step), report_name=report_name
        )
    st.session_state["eeg_emg_q_report"] = rep
    st.session_state["eeg_emg_q_path"] = selected_path
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

# ── Verdict + shape metrics ───────────────────────────────────────────────────
verdict = "PASS" if rep.passed else "FAIL"
color = "green" if rep.passed else "red"
st.markdown(f"### Verdict: :{color}[**{verdict}**]")
if rep.error:
    st.error(rep.error)

cols = st.columns(4)
cols[0].metric("EEG shape", str(rep.eeg_trials_shape or "—"))
cols[1].metric("EMG shape", str(rep.emg_trials_shape or "—"))
cols[2].metric("Sliding windows", rep.n_windows)
cols[3].metric("Min time (samples)", rep.min_time_samples)

detail = {
    "eeg_key": rep.eeg_key,
    "emg_key": rep.emg_key,
    "pre_windowed": rep.pre_windowed,
    "time_aligned": rep.time_aligned,
    "eeg_nan/inf": f"{rep.eeg_nan} / {rep.eeg_inf}",
    "emg_nan/inf": f"{rep.emg_nan} / {rep.emg_inf}",
    "flat EEG ch (trial 0)": rep.flat_eeg_channels,
    "flat EMG ch (trial 0)": rep.flat_emg_channels,
}
st.dataframe(pd.DataFrame([detail]), use_container_width=True)

if rep.warnings:
    st.subheader("Warnings")
    for w in rep.warnings:
        st.warning(w)

# ── Signal statistics (load raw arrays from disk) ─────────────────────────────
npz_path_for_stats: str | None = st.session_state.get("eeg_emg_q_path") or selected_path
if npz_path_for_stats and os.path.isfile(npz_path_for_stats):
    st.subheader("Signal statistics")
    try:
        with st.spinner("Loading arrays for signal analysis…"):
            eeg_trials, emg_trials, _ = prepare_eeg_emg_trials(npz_path_for_stats)

        # Per-channel stats (averaged over trials and time)
        def _channel_stats(arr: np.ndarray, prefix: str) -> pd.DataFrame:
            # arr: (N_trials, C, T)
            rows = []
            for ch in range(arr.shape[1]):
                flat = arr[:, ch, :].ravel()
                rows.append(
                    {
                        "Channel": f"{prefix} {ch}",
                        "Mean": round(float(flat.mean()), 5),
                        "Std": round(float(flat.std()), 5),
                        "Min": round(float(flat.min()), 5),
                        "Max": round(float(flat.max()), 5),
                        "Range": round(float(flat.max() - flat.min()), 5),
                    }
                )
            return pd.DataFrame(rows)

        col_eeg_stat, col_emg_stat = st.columns(2)
        with col_eeg_stat:
            st.markdown("**EEG channels**")
            st.dataframe(
                _channel_stats(eeg_trials, "EEG"), use_container_width=True, hide_index=True
            )
        with col_emg_stat:
            st.markdown("**EMG channels**")
            st.dataframe(
                _channel_stats(emg_trials, "EMG"), use_container_width=True, hide_index=True
            )

        # ── Cross-channel correlation heatmap (trial 0 mean over time) ────────
        st.subheader("EEG–EMG cross-channel correlation (trial 0)")
        trial_eeg = eeg_trials[0]  # (C_eeg, T)
        trial_emg = emg_trials[0]  # (C_emg, T)
        # Build combined matrix [EEG | EMG] (C_eeg+C_emg, T) then correlate
        combined = np.concatenate([trial_eeg, trial_emg], axis=0)  # (C, T)
        corr_mat = np.corrcoef(combined)  # (C, C)
        n_eeg = trial_eeg.shape[0]
        n_emg = trial_emg.shape[0]

        fig_h, ax_h = plt.subplots(
            figsize=(max(6, n_eeg + n_emg) * 0.6, max(5, n_eeg + n_emg) * 0.5)
        )
        im = ax_h.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        labels = [f"EEG{i}" for i in range(n_eeg)] + [f"EMG{i}" for i in range(n_emg)]
        ax_h.set_xticks(range(len(labels)))
        ax_h.set_yticks(range(len(labels)))
        ax_h.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax_h.set_yticklabels(labels, fontsize=8)
        # Highlight EEG/EMG boundary
        ax_h.axhline(n_eeg - 0.5, color="white", linewidth=1.5, linestyle="--")
        ax_h.axvline(n_eeg - 0.5, color="white", linewidth=1.5, linestyle="--")
        plt.colorbar(im, ax=ax_h, label="Pearson r")
        ax_h.set_title("Channel cross-correlation — dashed line separates EEG / EMG")
        fig_h.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

        # ── Power spectral density preview ────────────────────────────────────
        st.subheader("Power spectrum preview (trial 0)")
        fs = float(eeg_fs)
        from scipy.signal import welch

        col_psd_l, col_psd_r = st.columns(2)
        for col, arr_t, label_prefix, n_ch in [
            (col_psd_l, trial_eeg, "EEG", n_eeg),
            (col_psd_r, trial_emg, "EMG", n_emg),
        ]:
            with col:
                fig_p, ax_p = plt.subplots(figsize=(5.5, 3.5))
                for ch in range(n_ch):
                    f, Pxx = welch(arr_t[ch], fs=fs, nperseg=min(256, arr_t.shape[-1]))
                    ax_p.semilogy(f, Pxx, alpha=0.7, linewidth=0.9, label=f"CH{ch}")
                ax_p.set_xlabel("Frequency (Hz)")
                ax_p.set_ylabel("PSD (log scale)")
                ax_p.set_title(f"{label_prefix} — all channels")
                ax_p.legend(fontsize=6, ncol=2)
                ax_p.grid(alpha=0.3)
                fig_p.tight_layout()
                st.pyplot(fig_p)
                plt.close(fig_p)

    except Exception as exc:
        st.warning(f"Could not compute signal statistics: {exc}")
