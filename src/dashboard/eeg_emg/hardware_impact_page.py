# EEG–EMG hardware impact — Streamlit page

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
from src.dashboard.eeg_emg_torch_hardware import (
    get_torch_system_info,
    list_saved_torch_hardware_reports,
    load_torch_hardware_payload,
    run_torch_profile,
    save_torch_hardware_report,
)

st.set_page_config(
    page_title="EEG–EMG Hardware Impact | Umbra",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ EEG–EMG Hardware Impact (PyTorch)")
st.caption("Measure checkpoint load time and forward-pass latency on your validation windows.")

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

st.sidebar.header("Checkpoint")
if not os.path.isdir(EEG_EMG_MODEL_DIR):
    st.sidebar.error("Model directory missing.")
    pth_files: list[str] = []
else:
    pth_files = sorted(
        f for f in os.listdir(EEG_EMG_MODEL_DIR) if f.endswith(".pth") and not f.startswith(".")
    )

model_file: str | None = None
if pth_files:
    model_file = st.sidebar.selectbox("Model", pth_files)

st.sidebar.header("Profiling")
bs_str = st.sidebar.text_input("Batch sizes (comma-separated)", value="1, 4, 8, 16")
n_runs = st.sidebar.number_input("Batches per size", 1, 100, 5)
step = st.sidebar.number_input("Window step", 1, 2048, 128)
val_ratio = st.sidebar.slider("Validation fraction", 0.05, 0.5, 0.2, 0.05)
no_cuda = st.sidebar.checkbox("Force CPU", value=False)
run_label = st.sidebar.text_input("Report label", value="profile")

sys_info = get_torch_system_info()
st.sidebar.caption(
    f"Torch {sys_info.torch_version} · "
    f"{'CUDA ' + str(sys_info.cuda_version) if sys_info.cuda_available else 'CPU only'}"
)

if st.sidebar.button("Run profile", type="primary") and selected_npz and model_file:
    try:
        batch_sizes = [int(x.strip()) for x in bs_str.split(",") if x.strip()]
    except ValueError:
        batch_sizes = [1, 4, 8, 16]
    with st.spinner("Profiling…"):
        report = run_torch_profile(
            selected_npz,
            model_file,
            batch_sizes=batch_sizes,
            n_batch_runs=int(n_runs),
            step=int(step),
            val_ratio=float(val_ratio),
            no_cuda=no_cuda,
        )
    st.session_state["eeg_emg_hw"] = report
    try:
        p = save_torch_hardware_report(report, name=run_label)
        st.sidebar.success(f"Saved `{p.name}`")
    except Exception as exc:
        st.sidebar.warning(f"Save failed: {exc}")

st.sidebar.divider()
saved_hw = list_saved_torch_hardware_reports()
if saved_hw:
    lab = [f"{n} — {t}" for n, t, _ in saved_hw]
    ix = st.sidebar.selectbox("Saved reports", range(len(lab)), format_func=lambda i: lab[i])
    if st.sidebar.button("Load saved"):
        path = saved_hw[ix][2]
        payload = load_torch_hardware_payload(path)
        if payload:
            st.session_state["eeg_emg_hw"] = payload.get("report")

rep = st.session_state.get("eeg_emg_hw")
if rep is None:
    st.info("Select data + checkpoint and run profiling.")
    st.stop()

if isinstance(rep, dict):
    loading = rep.get("loading", {})
    inference = rep.get("inference", [])
else:
    loading = rep.loading
    inference = rep.inference

st.metric("Load time (s)", f"{loading.get('load_time_s', 0):.3f}")
if loading.get("load_error"):
    st.error(loading["load_error"])
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("RAM after load (MB)", f"{loading.get('ram_after_mb', 0):.1f}")
    cm = loading.get("cuda_mem_after_mb")
    if cm is not None:
        c2.metric("CUDA alloc (MB)", f"{cm:.1f}")
    c3.metric("File (MB)", f"{loading.get('file_size_mb', 0):.2f}")

if inference:
    st.subheader("Inference latency")
    st.dataframe(pd.DataFrame(inference), use_container_width=True)
else:
    st.warning("No inference timings (empty val set or error).")
