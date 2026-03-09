# Dataset quality checker (F6) – Streamlit page

import os
import sys
from pathlib import Path

# Ensure project root is on path when this page runs (e.g. streamlit run from dashboard)
_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
import pandas as pd
import altair as alt

from src.config import PREPROCESS_PATH
from src.dashboard.dataset_quality import (
    get_available_datasets,
    check_quality,
)
from src.emg_movement.gestures import label_to_gesture_name

st.set_page_config(
    page_title="Dataset Quality Checker | Umbra",
    page_icon="📋",
    layout="wide",
)
st.title("Dataset Quality Checker")
st.caption(
    "Validate preprocessed EMG datasets (X.npy / y.npy): integrity, label distribution, "
    "and train/val/test split readiness."
)

if not st.session_state.get("quality_report"):
    st.session_state["quality_report"] = None

# Sidebar: dataset selection
st.sidebar.header("Dataset")
if not os.path.isdir(PREPROCESS_PATH):
    st.sidebar.error("Preprocessed data directory not found.")
    available = []
else:
    available = get_available_datasets()

if not available:
    st.sidebar.warning("No preprocessed datasets found.")
    st.info("Run preprocessing first: `python -m src.main` then choose a dataset.")
    st.stop()

selected = st.sidebar.selectbox(
    "Choose a dataset",
    available,
    key="quality_dataset_select",
)
run_check = st.sidebar.button("Run quality check")

if run_check and selected:
    with st.spinner("Running quality checks..."):
        report = check_quality(selected)
    st.session_state["quality_report"] = report

report = st.session_state.get("quality_report")
if report is None:
    st.info("Select a dataset and click **Run quality check** to see the report.")
    st.stop()

# Report
st.header(f"Report: dataset `{report.dataset_id}`")

if not report.loaded:
    st.error(report.error or "Dataset could not be loaded.")
    st.stop()

# Verdict
if report.passed:
    st.success(
        "**Verdict: PASS** — No critical issues. Dataset is suitable for training."
    )
else:
    st.error(
        "**Verdict: FAIL** — Critical issues found (NaN/Inf, too few classes, or too few samples)."
    )

if report.warnings:
    st.warning("**Warnings:**")
    for w in report.warnings:
        st.markdown(f"- {w}")

# Basic stats
st.subheader("Basic stats")
c1, c2, c3, c4 = st.columns(4)
c1.metric("X shape", str(report.x_shape))
c2.metric("y shape", str(report.y_shape))
c3.metric("X dtype", report.x_dtype)
c4.metric("y dtype", report.y_dtype)

st.subheader("Integrity")
i1, i2, i3, i4 = st.columns(4)
i1.metric("NaN in X", report.x_nan_count)
i2.metric("Inf in X", report.x_inf_count)
i3.metric("NaN in y", report.y_nan_count)
i4.metric("Inf in y", report.y_inf_count)

# Labels
st.subheader("Label distribution")
st.write(f"**Number of classes:** {report.num_classes}")
st.write(f"**Rest label (0) count:** {report.rest_label_count}")
st.write(f"**Balance ratio (min/max per class):** {report.balance_ratio:.3f}")
st.write(f"**Min / max samples per class:** {report.min_count} / {report.max_count}")

if report.counts_per_class:
    df_counts = pd.DataFrame(
        [
            {"movement": label_to_gesture_name(k), "label": k, "count": v}
            for k, v in sorted(report.counts_per_class.items())
        ]
    )
    chart = (
        alt.Chart(df_counts)
        .mark_bar()
        .encode(
            x=alt.X(
                "movement:N", title="Movement", sort="-y", axis=alt.Axis(labelAngle=90)
            ),
            y=alt.Y("count:Q", title="Count"),
            tooltip=["movement", "label", "count"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)
    with st.expander("Counts per class (table)"):
        st.dataframe(df_counts[["movement", "count"]], use_container_width=True)

# Split simulation
if report.train_per_class is not None:
    st.subheader("Suggested split (70% / 15% / 15%)")
    t_train = sum(report.train_per_class.values())
    t_val = sum(report.val_per_class.values())
    t_test = sum(report.test_per_class.values())
    s1, s2, s3 = st.columns(3)
    s1.metric("Train samples", t_train)
    s2.metric("Val samples", t_val)
    s3.metric("Test samples", t_test)
    if (
        report.classes_with_few_train
        or report.classes_with_few_val
        or report.classes_with_few_test
    ):
        st.caption(
            "Some classes have very few samples in train/val/test; see warnings above."
        )

# Suspicious windows
st.subheader("Suspicious windows")
w1, w2 = st.columns(2)
w1.metric("Flat (low variance) windows", report.flat_window_count)
w2.metric("All-zero windows", report.zero_window_count)
st.caption(f"Total windows: {report.total_windows}")
