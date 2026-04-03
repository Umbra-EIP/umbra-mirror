# EEG–EMG model comparator — Streamlit page
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

from src.dashboard.eeg_emg_dataset_quality import list_npz_files
from src.dashboard.eeg_emg_model_compare import (
    comparison_result_from_saved_payload,
    list_saved_comparisons,
    load_comparison,
    run_eeg_emg_comparison,
    save_comparison,
)
from src.eeg_emg.dashboard_defaults import (
    EEG_EMG_DATA_DIR,
    EEG_EMG_DEFAULT_NPZ,
    EEG_EMG_MODEL_DIR,
)

st.set_page_config(
    page_title="EEG–EMG Model Comparator | Umbra",
    layout="wide",
)

st.title("EEG–EMG Model Comparator")
st.caption("Compare checkpoints: MSE, RMSE, MAE, R², Pearson r — all on the same validation split.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Data (.npz)")
npz_files = list_npz_files()
uploaded = st.sidebar.file_uploader("Override: upload .npz", type=["npz"])
selected_npz: str | None = None

default_npz_name = Path(EEG_EMG_DEFAULT_NPZ).name
npz_index = npz_files.index(default_npz_name) if default_npz_name in npz_files else 0
if uploaded is not None:
    dest = Path(EEG_EMG_DATA_DIR) / "_uploads" / uploaded.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(uploaded.getvalue())
    selected_npz = str(dest)
elif npz_files:
    selected_npz = os.path.join(
        EEG_EMG_DATA_DIR, st.sidebar.selectbox("Dataset", npz_files, index=npz_index)
    )

st.sidebar.header("Checkpoints")
pth_files: list[str] = []
if not os.path.isdir(EEG_EMG_MODEL_DIR):
    st.sidebar.error("Model directory missing.")
else:
    pth_files = sorted(
        f for f in os.listdir(EEG_EMG_MODEL_DIR) if f.endswith(".pth") and not f.startswith(".")
    )

# Pre-select all available PTH files by default
models = st.sidebar.multiselect(
    "Models to compare",
    options=pth_files,
    default=pth_files,
)

st.sidebar.header("Protocol")
step = st.sidebar.number_input("Window step", 1, 2048, 128)
val_ratio = st.sidebar.slider("Validation fraction", 0.05, 0.5, 0.2, 0.05)
normalize = st.sidebar.checkbox("Z-score normalize", value=True)
no_cuda = st.sidebar.checkbox("Force CPU", value=False)
seed = st.sidebar.number_input("RNG seed", 0, 2**31 - 1, 42)
run_name = st.sidebar.text_input("Run label (for save)", value="compare")

if st.sidebar.button("Run comparison", type="primary") and selected_npz and models:
    with st.spinner("Evaluating checkpoints…"):
        result = run_eeg_emg_comparison(
            selected_npz,
            models,
            step=int(step),
            val_ratio=float(val_ratio),
            normalize=normalize,
            seed=int(seed),
            no_cuda=no_cuda,
        )
    st.session_state["eeg_emg_cmp"] = result
    try:
        p = save_comparison(result, name=run_name)
        st.sidebar.success(f"Saved `{p.name}`")
    except Exception as exc:
        st.sidebar.warning(f"Save failed: {exc}")

st.sidebar.divider()
st.sidebar.subheader("Saved runs")
saved = list_saved_comparisons()
if saved:
    labels = [f"{n} — {t}" for n, t, _ in saved]
    idx = st.sidebar.selectbox("Load", range(len(labels)), format_func=lambda i: labels[i])
    if st.sidebar.button("Load saved"):
        path = saved[idx][2]
        data = load_comparison(path)
        if data:
            st.session_state["eeg_emg_cmp"] = comparison_result_from_saved_payload(data)

if not pth_files:
    st.warning(f"No `.pth` files in `{EEG_EMG_MODEL_DIR}/`.")

raw = st.session_state.get("eeg_emg_cmp")
if raw is None:
    st.info("Select a dataset and models, then click **Run comparison**.")
    st.stop()

# ── Build dataframe ───────────────────────────────────────────────────────────
rows = []
for row in raw.rows:
    rows.append(
        {
            "Model": row.model_file,
            "MSE": row.mse if not row.error else None,
            "RMSE": row.rmse if not row.error else None,
            "MAE": row.mae if not row.error else None,
            "R²": row.r2 if not row.error else None,
            "Mean Pearson r": row.mean_pearson if not row.error else None,
            "Best ckpt MSE": row.best_val_mse_in_checkpoint,
            "Error": row.error,
        }
    )
df = pd.DataFrame(rows)
good = df[df["Error"].isna()].copy()

# ── Comparison table ──────────────────────────────────────────────────────────
st.subheader("Results table")


def _colour_r(val: float | None) -> str:
    if val is None:
        return ""
    g = int(min(max(val, 0.0), 1.0) * 180)
    return f"background-color: rgba(0,{g},80,0.15)"


def _colour_err(val: float | None) -> str:
    if val is None:
        return ""
    # Lower is better: invert normalised value
    mn = df["MSE"].dropna().min() if not df["MSE"].dropna().empty else 0.0
    mx = df["MSE"].dropna().max() if not df["MSE"].dropna().empty else 1.0
    norm = 1.0 - (val - mn) / max(mx - mn, 1e-9)
    g = int(norm * 180)
    return f"background-color: rgba(0,{g},80,0.15)"


styled = df.style.applymap(_colour_r, subset=["R²", "Mean Pearson r"])
if not df["MSE"].dropna().empty:
    styled = styled.applymap(_colour_err, subset=["MSE", "RMSE", "MAE"])
st.dataframe(
    styled.format(
        {
            "MSE": "{:.5f}",
            "RMSE": "{:.5f}",
            "MAE": "{:.5f}",
            "R²": "{:.4f}",
            "Mean Pearson r": "{:.4f}",
            "Best ckpt MSE": lambda v: f"{v:.5f}" if v is not None else "—",
        },
        na_rep="—",
    ),
    use_container_width=True,
    hide_index=True,
)

# ── Bar charts ────────────────────────────────────────────────────────────────
if not good.empty:
    st.subheader("Visual comparison")
    metrics_to_plot = [
        ("MSE (lower is better)", "MSE", True),
        ("R² (higher is better)", "R²", False),
        ("Mean Pearson r (higher is better)", "Mean Pearson r", False),
    ]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 4))
    names = good["Model"].tolist()
    x = np.arange(len(names))

    for ax, (title, col, lower_is_better) in zip(np.atleast_1d(axes), metrics_to_plot):
        vals = good[col].tolist()
        colors = []
        for v in vals:
            if lower_is_better:
                mn, mx = min(vals), max(vals)
                norm = 1.0 - (v - mn) / max(mx - mn, 1e-9)
            else:
                mn, mx = min(vals), max(vals)
                norm = (v - mn) / max(mx - mn, 1e-9)
            g = int(norm * 200)
            colors.append((0, g / 255, 80 / 255, 0.8))
        bars = ax.bar(x, vals, color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
