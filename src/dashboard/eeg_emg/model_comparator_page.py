# EEG–EMG model comparator — Streamlit page

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st

from src.config import EEG_EMG_DATA_DIR, EEG_EMG_MODEL_DIR
from src.dashboard.eeg_emg_dataset_quality import list_npz_files
from src.dashboard.eeg_emg_model_compare import (
    comparison_result_from_saved_payload,
    list_saved_comparisons,
    load_comparison,
    run_eeg_emg_comparison,
    save_comparison,
)

st.set_page_config(
    page_title="EEG–EMG Model Comparator | Umbra",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 EEG–EMG Model Comparator")
st.caption(
    "Run the same validation protocol on several `.pth` checkpoints (MSE / R² on the val split)."
)

st.sidebar.header("Data (.npz)")
npz_files = list_npz_files()
uploaded = st.sidebar.file_uploader("Or upload .npz", type=["npz"])
selected_npz: str | None = None
if uploaded is not None:
    dest = Path(EEG_EMG_DATA_DIR) / "_uploads" / uploaded.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(uploaded.getvalue())
    selected_npz = str(dest)
elif npz_files:
    selected_npz = os.path.join(EEG_EMG_DATA_DIR, st.sidebar.selectbox("Dataset", npz_files))

st.sidebar.header("Checkpoints")
pth_files: list[str] = []
if not os.path.isdir(EEG_EMG_MODEL_DIR):
    st.sidebar.error("Model directory missing.")
else:
    pth_files = sorted(
        f for f in os.listdir(EEG_EMG_MODEL_DIR) if f.endswith(".pth") and not f.startswith(".")
    )

st.sidebar.header("Protocol")
step = st.sidebar.number_input("Window step", 1, 2048, 128)
val_ratio = st.sidebar.slider("Validation fraction", 0.05, 0.5, 0.2, 0.05)
normalize = st.sidebar.checkbox("Z-score normalize", value=True)
no_cuda = st.sidebar.checkbox("Force CPU", value=False)
seed = st.sidebar.number_input("RNG seed", 0, 2**31 - 1, 42)

run_name = st.sidebar.text_input("Run label (for save)", value="compare")

models = st.sidebar.multiselect(
    "Models to compare",
    options=pth_files,
    default=pth_files[: min(3, len(pth_files))],
)

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
    st.info("Select a dataset and models, then run comparison.")
    st.stop()

rows = []
for row in raw.rows:
    rows.append(
        {
            "model": row.model_file,
            "MSE": row.mse if not row.error else None,
            "R²": row.r2 if not row.error else None,
            "best_val_mse (ckpt)": row.best_val_mse_in_checkpoint,
            "error": row.error,
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True)
