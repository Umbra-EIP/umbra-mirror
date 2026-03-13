# Dataset quality checker (F6) – Streamlit page

import os
import sys
from io import StringIO
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import altair as alt
import pandas as pd
import streamlit as st

from src.config import PREPROCESS_PATH
from src.dashboard.dataset_quality import (
    MIN_CLASS_BALANCE_RATIO,
    MIN_SAMPLES_PER_CLASS_TEST,
    MIN_SAMPLES_PER_CLASS_TRAIN,
    MIN_SAMPLES_PER_CLASS_VAL,
    QualityThresholds,
    check_quality,
    get_available_datasets,
    list_saved_reports,
    load_report,
    save_report,
)
from src.emg_movement.gestures import label_to_gesture_name

st.set_page_config(
    page_title="Dataset Quality Checker | Umbra",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom spacing and typography
st.markdown(
    """
    <style>
    .metric-card { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 1rem 1.25rem; border-radius: 0.5rem; margin-bottom: 0.5rem; }
    .verdict-pass { background: linear-gradient(135deg, #065f46 0%, #047857 100%); color: white; padding: 1.25rem 1.5rem; border-radius: 0.5rem; font-weight: 600; }
    .verdict-fail { background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); color: white; padding: 1.25rem 1.5rem; border-radius: 0.5rem; font-weight: 600; }
    .section-title { font-size: 1.1rem; font-weight: 600; color: #94a3b8; margin-top: 1.5rem; margin-bottom: 0.75rem; }
    .movement-list { font-family: ui-monospace, monospace; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Dataset Quality Checker")
st.caption(
    "Validate preprocessed EMG datasets (X.npy / y.npy): integrity, label distribution, "
    "and train/val/test split readiness."
)

if not st.session_state.get("quality_report"):
    st.session_state["quality_report"] = None

# ----- Sidebar -----
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

# Clear report when user switches to a different dataset
if (
    st.session_state.get("quality_report") is not None
    and st.session_state["quality_report"].dataset_id != selected
):
    st.session_state["quality_report"] = None

st.sidebar.divider()
st.sidebar.subheader("Thresholds")
st.sidebar.caption("Minimum samples per class in each split (leave 0 for default).")
min_train = st.sidebar.number_input(
    "Min samples (train)",
    min_value=0,
    value=MIN_SAMPLES_PER_CLASS_TRAIN,
    step=1,
    help="Default: 10",
)
min_val = st.sidebar.number_input(
    "Min samples (val)",
    min_value=0,
    value=MIN_SAMPLES_PER_CLASS_VAL,
    step=1,
    help="Default: 2",
)
min_test = st.sidebar.number_input(
    "Min samples (test)",
    min_value=0,
    value=MIN_SAMPLES_PER_CLASS_TEST,
    step=1,
    help="Default: 2",
)
min_balance = st.sidebar.slider(
    "Min balance ratio (min/max per class)",
    min_value=0.0,
    max_value=0.5,
    value=float(MIN_CLASS_BALANCE_RATIO),
    step=0.01,
    help="Default: 0.05",
)

thresholds = QualityThresholds(
    min_samples_train=min_train if min_train > 0 else None,
    min_samples_val=min_val if min_val > 0 else None,
    min_samples_test=min_test if min_test > 0 else None,
    min_balance_ratio=min_balance,
)

st.sidebar.divider()
st.sidebar.subheader("Report name")
st.sidebar.caption("Give this report a name to save it separately (e.g. 'strict', 'loose').")
report_name = st.sidebar.text_input(
    "Report name",
    value="default",
    key="report_name_input",
    help="Only letters, numbers, hyphens and underscores. Used as part of the saved filename.",
)
report_name = report_name.strip() or "default"

run_check = st.sidebar.button("Run quality check", type="primary")

if run_check and selected:
    with st.spinner("Running quality checks..."):
        report = check_quality(selected, thresholds=thresholds, report_name=report_name)
    st.session_state["quality_report"] = report
    try:
        path = save_report(report)
        st.sidebar.success(f"Saved as **{path.name}**")
    except Exception as e:
        st.sidebar.warning(f"Report displayed but save failed: {e}")

# ----- Saved reports -----
st.sidebar.divider()
st.sidebar.subheader("Saved reports")
saved = list_saved_reports()
if saved:
    saved_options = [f"Dataset {did} — {name}" for did, name, _ in saved]
    saved_paths = [path for _, _, path in saved]
    chosen = st.sidebar.selectbox(
        "Load a saved report",
        options=range(len(saved_options)),
        format_func=lambda i: saved_options[i],
        key="saved_report_select",
    )
    col_load, col_del = st.sidebar.columns(2)
    if col_load.button("Load", key="btn_load_report"):
        report_loaded = load_report(saved_paths[chosen])
        if report_loaded is not None:
            st.session_state["quality_report"] = report_loaded
        else:
            st.sidebar.error("Failed to load report.")
    if col_del.button("Delete", key="btn_delete_report"):
        try:
            saved_paths[chosen].unlink()
            st.sidebar.success("Report deleted.")
            st.session_state["quality_report"] = None
        except Exception as e:
            st.sidebar.error(f"Delete failed: {e}")
else:
    st.sidebar.caption("_No saved reports yet. Run a check to save._")

# Show which report is currently displayed
if st.session_state.get("quality_report") is not None:
    r = st.session_state["quality_report"]
    st.sidebar.caption(f"Viewing: dataset **{r.dataset_id}** — **{r.report_name}**")
else:
    st.sidebar.caption("_No report loaded_")

report = st.session_state.get("quality_report")
if report is None:
    st.info(
        "Select a dataset and click **Run quality check**, or load a saved report from the sidebar."
    )
    st.stop()

# ----- Report -----
# One-line summary
summary = (
    f"Dataset **{report.dataset_id}** — "
    f"report **{report.report_name}** — "
    f"{report.total_windows:,} windows, {report.num_classes} classes — "
    f"**{'PASS' if report.passed else 'FAIL'}**"
)
st.markdown(summary)

st.divider()

# Verdict
st.subheader("Verdict")
if report.loaded:
    if report.passed:
        st.markdown(
            '<p class="verdict-pass">✓ PASS — No critical issues. Dataset is suitable for training.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="verdict-fail">✗ FAIL — Critical issues found (NaN/Inf, too few classes, or too few samples).</p>',
            unsafe_allow_html=True,
        )
else:
    st.error(report.error or "Dataset could not be loaded.")
    st.stop()

# Warnings (with movement names for under-represented)
if report.warnings:
    st.markdown("**Warnings**")
    for w in report.warnings:
        if "samples in train:" in w or "samples in val:" in w or "samples in test:" in w:
            continue
        st.markdown(f"- {w}")

    # Under-represented movements (with actual names)
    if report.classes_with_few_train or report.classes_with_few_val or report.classes_with_few_test:
        st.markdown("**Under-represented movements**")
        if report.classes_with_few_train:
            names = [label_to_gesture_name(c) for c in report.classes_with_few_train]
            st.markdown(f"- **Train:** {', '.join(names)}")
        if report.classes_with_few_val:
            names = [label_to_gesture_name(c) for c in report.classes_with_few_val]
            st.markdown(f"- **Val:** {', '.join(names)}")
        if report.classes_with_few_test:
            names = [label_to_gesture_name(c) for c in report.classes_with_few_test]
            st.markdown(f"- **Test:** {', '.join(names)}")
else:
    st.success("No warnings.")

st.divider()

# Thresholds used for this report
st.subheader("Thresholds used")
if report.thresholds_used:
    t = report.thresholds_used
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Min train / class", t.get("min_samples_train", "—"))
    tc2.metric("Min val / class", t.get("min_samples_val", "—"))
    tc3.metric("Min test / class", t.get("min_samples_test", "—"))
    tc4.metric("Min balance ratio", f"{t.get('min_balance_ratio', 0):.3f}")
else:
    st.caption("No threshold information saved (older report).")

st.divider()

# Basic stats + X value range
st.subheader("Data overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("X shape", str(report.x_shape))
col2.metric("y shape", str(report.y_shape))
col3.metric("X dtype", report.x_dtype)
col4.metric("y dtype", report.y_dtype)

if report.x_min is not None and report.x_max is not None and report.x_mean is not None:
    st.markdown("**X value range**")
    rx1, rx2, rx3 = st.columns(3)
    rx1.metric("Min", f"{report.x_min:.4f}")
    rx2.metric("Max", f"{report.x_max:.4f}")
    rx3.metric("Mean", f"{report.x_mean:.4f}")
    if report.x_mean_per_channel:
        with st.expander("Mean per channel"):
            ch_df = pd.DataFrame(
                {
                    "channel": range(len(report.x_mean_per_channel)),
                    "mean": report.x_mean_per_channel,
                }
            )
            st.dataframe(ch_df, use_container_width=True)

st.divider()

# Integrity
st.subheader("Integrity")
integrity_ok = (
    report.x_nan_count == 0
    and report.x_inf_count == 0
    and report.y_nan_count == 0
    and report.y_inf_count == 0
)
if integrity_ok:
    st.caption("All clean (no NaN/Inf).")
else:
    st.caption("Issues detected.")
i1, i2, i3, i4 = st.columns(4)
i1.metric("NaN in X", report.x_nan_count)
i2.metric("Inf in X", report.x_inf_count)
i3.metric("NaN in y", report.y_nan_count)
i4.metric("Inf in y", report.y_inf_count)

st.divider()

# Label distribution
st.subheader("Label distribution")
lb1, lb2, lb3 = st.columns(3)
lb1.metric("Number of classes", report.num_classes)
lb2.metric("Rest (label 0) count", report.rest_label_count)
lb3.metric("Balance ratio", f"{report.balance_ratio:.3f}")
st.caption(f"Min / max samples per class: {report.min_count} / {report.max_count}")

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
            x=alt.X("movement:N", title="Movement", sort="-y", axis=alt.Axis(labelAngle=90)),
            y=alt.Y("count:Q", title="Count"),
            tooltip=["movement", "label", "count"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    with st.expander("Counts per class (table)"):
        st.dataframe(df_counts[["movement", "count"]], use_container_width=True)

st.divider()

# Suggested split + per-class table
if report.train_per_class is not None:
    st.subheader("Suggested split (70% / 15% / 15%)")
    t_train = sum(report.train_per_class.values())
    t_val = sum(report.val_per_class.values())
    t_test = sum(report.test_per_class.values())
    s1, s2, s3 = st.columns(3)
    s1.metric("Train samples", t_train)
    s2.metric("Val samples", t_val)
    s3.metric("Test samples", t_test)

    st.markdown("**Per-class split**")
    split_df = pd.DataFrame(
        [
            {
                "Movement": label_to_gesture_name(lab),
                "Train": report.train_per_class[int(lab)],
                "Val": report.val_per_class[int(lab)],
                "Test": report.test_per_class[int(lab)],
            }
            for lab in sorted(report.train_per_class.keys())
        ]
    )
    st.dataframe(split_df, use_container_width=True, height=300)

    st.info(
        "Split is simulated with a fixed random seed (stratified). "
        "If your data contains multiple subjects or sessions, consider splitting by subject to avoid leakage."
    )

st.divider()

# Suspicious windows
st.subheader("Suspicious windows")
w1, w2, w3 = st.columns(3)
w1.metric("Flat (low variance)", report.flat_window_count)
w2.metric("All-zero", report.zero_window_count)
w3.metric("Total windows", report.total_windows)

st.divider()

# Export
st.subheader("Export report")
buf = StringIO()
buf.write(f"Dataset Quality Report — dataset_id={report.dataset_id} — name={report.report_name}\n")
buf.write("=" * 50 + "\n")
buf.write(f"Verdict: {'PASS' if report.passed else 'FAIL'}\n")
if report.thresholds_used:
    t = report.thresholds_used
    buf.write(
        f"Thresholds: min_train={t.get('min_samples_train')} "
        f"min_val={t.get('min_samples_val')} "
        f"min_test={t.get('min_samples_test')} "
        f"min_balance_ratio={t.get('min_balance_ratio')}\n"
    )
buf.write(f"X shape: {report.x_shape}, y shape: {report.y_shape}\n")
buf.write(f"Total windows: {report.total_windows}, Classes: {report.num_classes}\n")
if report.x_min is not None:
    buf.write(
        f"X range: min={report.x_min:.4f}, max={report.x_max:.4f}, mean={report.x_mean:.4f}\n"
    )
buf.write(
    f"NaN/Inf in X: {report.x_nan_count}/{report.x_inf_count}, in y: {report.y_nan_count}/{report.y_inf_count}\n"
)
buf.write(f"Balance ratio: {report.balance_ratio:.3f}\n")
buf.write("Warnings:\n")
for w in report.warnings:
    buf.write(f"  - {w}\n")
if report.counts_per_class:
    buf.write("\nCounts per class:\n")
    for k, v in sorted(report.counts_per_class.items()):
        buf.write(f"  {label_to_gesture_name(k)}: {v}\n")

txt_report = buf.getvalue()
ex1, ex2 = st.columns(2)
with ex1:
    slug = f"dataset_{report.dataset_id}_{report.report_name}"
    st.download_button(
        "Download report (text)",
        data=txt_report,
        file_name=f"quality_report_{slug}.txt",
        mime="text/plain",
    )
with ex2:
    if report.counts_per_class:
        df_export = pd.DataFrame(
            [
                {"movement": label_to_gesture_name(k), "label": k, "count": v}
                for k, v in sorted(report.counts_per_class.items())
            ]
        )
        st.download_button(
            "Download counts (CSV)",
            data=df_export.to_csv(index=False),
            file_name=f"quality_counts_{slug}.csv",
            mime="text/csv",
        )

if report.train_per_class is not None:
    split_export = pd.DataFrame(
        [
            {
                "movement": label_to_gesture_name(lab),
                "train": report.train_per_class[int(lab)],
                "val": report.val_per_class[int(lab)],
                "test": report.test_per_class[int(lab)],
            }
            for lab in sorted(report.train_per_class.keys())
        ]
    )
    st.download_button(
        "Download split table (CSV)",
        data=split_export.to_csv(index=False),
        file_name=f"quality_split_{slug}.csv",
        mime="text/csv",
    )
