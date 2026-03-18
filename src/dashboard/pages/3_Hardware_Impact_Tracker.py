# Hardware Impact Tracker — Streamlit page

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard.hardware_profiler import (
    HardwareReport,
    get_available_datasets,
    get_available_models,
    list_saved_reports,
    load_report,
    run_hardware_profile,
    save_report,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hardware Impact Tracker | Umbra",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.block-container { padding-top: 1.5rem; }

.section-title {
    font-size: 0.7rem;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.25rem;
    margin-bottom: 0.75rem;
}

.info-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(148,163,184,0.1);
    border-left: 3px solid #334155;
    border-radius: 0.65rem;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.55rem;
}
.info-card-accent  { border-left-color: #0ea5e9; }
.info-card-warn    { border-left-color: #f59e0b; }
.info-card-success { border-left-color: #059669; }
.info-card-danger  { border-left-color: #dc2626; }

.card-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.25rem;
}
.card-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
}
.card-sub {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.15rem;
}

.badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    letter-spacing: 0.04em;
}
.badge-gpu  { background: linear-gradient(135deg,#1e3a5f,#1d4ed8); color:#fff; }
.badge-cpu  { background: linear-gradient(135deg,#1a2e1a,#15803d); color:#fff; }
.badge-warn { background: linear-gradient(135deg,#78350f,#d97706); color:#fff; }

.empty-state { text-align:center; padding:4rem 1rem 3rem; color:#475569; }
.empty-icon  { font-size:3.5rem; margin-bottom:0.75rem; }
.empty-title { font-size:1.15rem; font-weight:600; color:#94a3b8; }
.empty-body  { font-size:0.9rem; margin-top:0.4rem; }

.synth-note {
    font-size: 0.78rem;
    color: #f59e0b;
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 0.4rem;
    padding: 0.4rem 0.7rem;
    margin-bottom: 0.75rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "hw_report" not in st.session_state:
    st.session_state["hw_report"] = None

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Model
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Model")
available_models = get_available_models()
if not available_models:
    st.sidebar.error("No .keras models found in src/models/")
    st.error("No models found. Train a model first.")
    st.stop()

selected_model: str = st.sidebar.selectbox(
    "Model to profile",
    options=available_models,
    key="hw_model_sel",
)

# ── Dataset ─────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("Data source")
st.sidebar.caption("Real dataset → inference on true EMG windows. None → synthetic noise.")

available_datasets = get_available_datasets()
use_dataset = st.sidebar.toggle(
    "Use real dataset",
    value=bool(available_datasets),
    key="hw_use_dataset",
)
selected_dataset: str | None = None

if use_dataset:
    if not available_datasets:
        st.sidebar.warning("No preprocessed datasets found. Synthetic data will be used.")
    else:
        selected_dataset = st.sidebar.selectbox(
            "Dataset", options=available_datasets, key="hw_dataset_sel"
        )
        n_windows: int = st.sidebar.slider(
            "Dataset windows",
            min_value=50,
            max_value=2000,
            value=500,
            step=50,
            help="Number of windows randomly sampled (seed 42) for profiling.",
        )
else:
    n_windows = 500  # unused but keeps variable defined

# ── Batch inference settings ─────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("Batch inference")

_all_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
selected_batch_sizes: list[int] = st.sidebar.multiselect(
    "Batch sizes to test",
    options=_all_batch_sizes,
    default=[1, 4, 8, 16, 32, 64],
    key="hw_batch_sizes",
)
if not selected_batch_sizes:
    selected_batch_sizes = [1, 8, 32]

n_batch_runs: int = st.sidebar.slider(
    "Timing runs per batch size",
    min_value=3,
    max_value=20,
    value=5,
    step=1,
    help="Number of predict() calls averaged per batch size.",
)

# ── Stress test settings ─────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("Stress test")

run_stress: bool = st.sidebar.toggle("Enable stress test", value=False, key="hw_stress")

if run_stress:
    n_stress: int = st.sidebar.slider(
        "Number of inferences",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Total batched predict() calls for the stress test.",
    )
    stress_bs: int = st.sidebar.selectbox(
        "Stress batch size",
        options=_all_batch_sizes,
        index=0,
        key="hw_stress_bs",
    )
else:
    n_stress = 500
    stress_bs = 1

# ── Saved reports ────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("Saved reports")

saved_reports = list_saved_reports()
if saved_reports:
    rep_labels = [f"{name}  ({saved_at[:16]})" for name, saved_at, _ in saved_reports]
    rep_paths = [p for _, _, p in saved_reports]
    chosen_idx: int = st.sidebar.selectbox(
        "Load a saved report",
        options=range(len(rep_labels)),
        format_func=lambda i: rep_labels[i],
        key="hw_saved_sel",
    )
    _r_load, _r_del = st.sidebar.columns(2)
    if _r_load.button("Load", use_container_width=True, key="hw_load_btn"):
        try:
            st.session_state["hw_report"] = load_report(rep_paths[chosen_idx])
            st.sidebar.success("Loaded.")
        except Exception as exc:
            st.sidebar.error(f"Load failed: {exc}")
    if _r_del.button("Delete", use_container_width=True, key="hw_del_btn"):
        try:
            rep_paths[chosen_idx].unlink()
            st.sidebar.success("Deleted.")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Delete failed: {exc}")
else:
    st.sidebar.caption("_No saved reports yet._")

# ── Run button ───────────────────────────────────────────────────────────────
st.sidebar.divider()
run_btn = st.sidebar.button(
    "▶  Run Profile",
    type="primary",
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚡ Hardware Impact Tracker")
st.caption(
    "Profile a Keras model's resource footprint — RAM & GPU memory at load time, "
    "inference latency across batch sizes, throughput, and peak memory under sustained load."
)

# ─────────────────────────────────────────────────────────────────────────────
# Run profiling
# ─────────────────────────────────────────────────────────────────────────────
report: HardwareReport | None = None

if run_btn:
    _status = st.empty()
    _bar = st.progress(0)

    def _cb(label: str, done: int, total: int) -> None:
        _bar.progress(done / max(total, 1))
        _status.caption(f"⏳ {label}")

    with st.spinner(""):
        report = run_hardware_profile(
            model_name=selected_model,
            dataset_id=selected_dataset if use_dataset else None,
            n_dataset_windows=n_windows,
            batch_sizes=sorted(selected_batch_sizes),
            n_batch_runs=n_batch_runs,
            run_stress=run_stress,
            n_stress_inferences=n_stress,
            stress_batch_size=stress_bs,
            progress_callback=_cb,
        )

    _bar.empty()
    _status.empty()
    st.session_state["hw_report"] = report

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
report = st.session_state.get("hw_report")

if report is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">⚡</div>
            <div class="empty-title">No profile run yet</div>
            <div class="empty-body">
                Select a model in the sidebar and click <strong>▶ Run Profile</strong>,
                or load a saved report.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_bar_props = {"cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4}
_cfg = report.config
_sys = report.system_info
_lp = report.loading_profile or {}
_bi = report.batch_inference  # list[dict]
_st = report.stress_test  # dict | None
_used_synth = _cfg.get("used_synthetic_data", False)
_has_gpu = _sys.get("has_gpu", False)


def _fmt_mb(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:,.1f} MB"


def _fmt_ms(v: float | None, decimals: int = 2) -> str:
    return f"{v:.{decimals}f} ms" if v is not None else "—"


def _fmt_params(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


# Synthetic data notice
if _used_synth:
    st.markdown(
        '<div class="synth-note">⚠️ <strong>Synthetic data</strong> — '
        "no real dataset was used. Latency measurements are accurate; "
        "confidence/class-level metrics are not applicable.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 1 · System Info
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">System Information</p>', unsafe_allow_html=True)

gpu_devices = _sys.get("gpu_devices", [])
gpu_label = ", ".join(g["name"] for g in gpu_devices) if gpu_devices else "None detected"
gpu_badge = (
    '<span class="badge badge-gpu">GPU</span>'
    if _has_gpu
    else '<span class="badge badge-cpu">CPU only</span>'
)

si1, si2, si3, si4, si5 = st.columns(5)
si1.metric("Platform", _sys.get("platform_str", "—")[:30])
si2.metric(
    "CPU cores",
    f"{_sys.get('cpu_count_physical', '?')} phys / {_sys.get('cpu_count_logical', '?')} logical",
)
si3.metric("RAM (total)", _fmt_mb(_sys.get("ram_total_mb")))
si4.metric("RAM (free at start)", _fmt_mb(_sys.get("ram_available_mb")))
si5.metric("TensorFlow", _sys.get("tensorflow_version", "—"))

st.markdown(
    f"**GPU:** {gpu_label} &nbsp; {gpu_badge} &nbsp;&nbsp; "
    f"**Python:** {_sys.get('python_version', '—')}",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2 · Loading Profile
# ─────────────────────────────────────────────────────────────────────────────
if _lp:
    st.divider()
    st.markdown('<p class="section-title">Model Loading Profile</p>', unsafe_allow_html=True)

    lp_loaded = _lp.get("loaded", False)
    if not lp_loaded:
        st.error(f"Model failed to load: {_lp.get('load_error', 'unknown error')}")
    else:
        l1, l2, l3, l4, l5 = st.columns(5)
        l1.metric("Load time", f"{_lp.get('load_time_s', 0):.3f} s")
        l2.metric("File size", _fmt_mb(_lp.get("file_size_mb")))
        l3.metric("Parameters", _fmt_params(_lp.get("param_count")))
        l4.metric(
            "RAM delta (load)",
            _fmt_mb(_lp.get("ram_delta_mb")),
            help="Process RSS after load minus before load. "
            "Includes Python/TF overhead; may differ from raw model size.",
        )
        l5.metric(
            "GPU mem delta",
            _fmt_mb(_lp.get("gpu_mem_delta_mb")) if _has_gpu else "N/A",
        )

        with st.expander("Full loading detail", expanded=False):
            rows = [
                ("Model", _lp.get("model_name", "—")),
                ("Input shape", str(_lp.get("model_input_shape", "—"))),
                ("Last modified", _lp.get("file_modified") or "—"),
                ("File size (MB)", f"{_lp.get('file_size_mb', 0):.3f}"),
                ("Parameters", _fmt_params(_lp.get("param_count"))),
                ("Load time (s)", f"{_lp.get('load_time_s', 0):.4f}"),
                ("RAM before (MB)", f"{_lp.get('ram_before_mb', 0):.1f}"),
                ("RAM after (MB)", f"{_lp.get('ram_after_mb', 0):.1f}"),
                ("RAM delta (MB)", f"{_lp.get('ram_delta_mb', 0):+.1f}"),
            ]
            if _has_gpu:
                rows += [
                    (
                        "GPU mem before (MB)",
                        f"{_lp.get('gpu_mem_before_mb', 0):.1f}"
                        if _lp.get("gpu_mem_before_mb") is not None
                        else "—",
                    ),
                    (
                        "GPU mem after (MB)",
                        f"{_lp.get('gpu_mem_after_mb', 0):.1f}"
                        if _lp.get("gpu_mem_after_mb") is not None
                        else "—",
                    ),
                    (
                        "GPU mem delta (MB)",
                        f"{_lp.get('gpu_mem_delta_mb', 0):+.1f}"
                        if _lp.get("gpu_mem_delta_mb") is not None
                        else "—",
                    ),
                ]
            df_lp = pd.DataFrame(rows, columns=["Metric", "Value"])
            st.dataframe(df_lp, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3 · Batch Inference Profile
# ─────────────────────────────────────────────────────────────────────────────
if _bi:
    st.divider()
    st.markdown('<p class="section-title">Batch Inference Profile</p>', unsafe_allow_html=True)

    data_note = (
        f"dataset **{report.dataset_id}** ({report.dataset_shape})"
        if report.dataset_id
        else "**synthetic data**"
    )
    st.caption(
        f"Measured on {data_note} · "
        f"{_cfg.get('n_batch_runs', 5)} timing runs per batch size · "
        "one warm-up call excluded."
    )

    df_bi = pd.DataFrame(
        [
            {
                "Batch size": b["batch_size"],
                "Mean (ms)": round(b["mean_latency_ms"], 2),
                "Std (ms)": round(b["std_latency_ms"], 2),
                "p50 (ms)": round(b["p50_ms"], 2),
                "p95 (ms)": round(b["p95_ms"], 2),
                "p99 (ms)": round(b["p99_ms"], 2),
                "Per sample (ms)": round(b["per_sample_ms"], 3),
                "Throughput (samp/s)": round(b["throughput_samples_per_sec"], 1),
            }
            for b in _bi
        ]
    )

    tab_table, tab_latency, tab_throughput = st.tabs(
        ["📋 Table", "⚡ Latency vs Batch", "🚀 Throughput"]
    )

    with tab_table:
        st.dataframe(
            df_bi,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Batch size": st.column_config.NumberColumn("Batch size", format="%d"),
                "Mean (ms)": st.column_config.NumberColumn("Mean (ms)", format="%.2f"),
                "Std (ms)": st.column_config.NumberColumn("Std (ms)", format="%.2f"),
                "p50 (ms)": st.column_config.NumberColumn("p50 (ms)", format="%.2f"),
                "p95 (ms)": st.column_config.NumberColumn("p95 (ms)", format="%.2f"),
                "p99 (ms)": st.column_config.NumberColumn("p99 (ms)", format="%.2f"),
                "Per sample (ms)": st.column_config.NumberColumn("Per sample (ms)", format="%.3f"),
                "Throughput (samp/s)": st.column_config.ProgressColumn(
                    "Throughput (samp/s)",
                    min_value=0,
                    max_value=float(df_bi["Throughput (samp/s)"].max()),
                    format="%.1f",
                ),
            },
        )

    with tab_latency:
        # Long-format for grouped bars: Mean / p50 / p95 / p99
        lat_long = []
        for b in _bi:
            bs = b["batch_size"]
            for metric, val in [
                ("Mean", b["mean_latency_ms"]),
                ("p50", b["p50_ms"]),
                ("p95", b["p95_ms"]),
                ("p99", b["p99_ms"]),
            ]:
                lat_long.append({"Batch size": str(bs), "Metric": metric, "ms": val})
        df_lat_long = pd.DataFrame(lat_long)

        chart_lat = (
            alt.Chart(df_lat_long)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "Batch size:O",
                    sort=[str(b["batch_size"]) for b in _bi],
                    title="Batch size",
                ),
                y=alt.Y("ms:Q", title="Latency (ms)"),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(scheme="tableau10"),
                    sort=["Mean", "p50", "p95", "p99"],
                ),
                tooltip=["Batch size", "Metric", alt.Tooltip("ms:Q", format=".2f")],
            )
            .properties(height=360, title="Latency vs batch size (lower = better)")
        )

        # Error band: mean ± std
        err_rows = [
            {
                "Batch size": str(b["batch_size"]),
                "lo": max(0.0, b["mean_latency_ms"] - b["std_latency_ms"]),
                "hi": b["mean_latency_ms"] + b["std_latency_ms"],
            }
            for b in _bi
        ]
        df_err = pd.DataFrame(err_rows)
        err_band = (
            alt.Chart(df_err)
            .mark_area(opacity=0.15, color="#60a5fa")
            .encode(
                x=alt.X("Batch size:O", sort=[str(b["batch_size"]) for b in _bi]),
                y=alt.Y("lo:Q"),
                y2=alt.Y2("hi:Q"),
            )
        )
        st.altair_chart(err_band + chart_lat, use_container_width=True)
        st.caption("Shaded band = mean ± 1σ across timing runs.")

        # Per-sample cost table
        st.markdown("**Per-sample latency** (batch total ÷ batch size):")
        df_per_sample = df_bi[["Batch size", "Per sample (ms)"]].copy()
        best_bs = int(df_bi.loc[df_bi["Per sample (ms)"].idxmin(), "Batch size"])
        st.dataframe(
            df_per_sample,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Most efficient batch size: **{best_bs}** "
            f"({df_bi.loc[df_bi['Batch size'] == best_bs, 'Per sample (ms)'].values[0]:.3f} ms/sample)."
        )

    with tab_throughput:
        df_tp = df_bi[["Batch size", "Throughput (samp/s)"]].copy()
        df_tp["Batch size"] = df_tp["Batch size"].astype(str)

        chart_tp = (
            alt.Chart(df_tp)
            .mark_bar(**_bar_props)
            .encode(
                x=alt.X(
                    "Batch size:O",
                    sort=[str(b["batch_size"]) for b in _bi],
                    title="Batch size",
                ),
                y=alt.Y("Throughput (samp/s):Q", title="Samples / second"),
                color=alt.Color(
                    "Throughput (samp/s):Q",
                    scale=alt.Scale(scheme="greens"),
                    legend=None,
                ),
                tooltip=[
                    "Batch size",
                    alt.Tooltip("Throughput (samp/s):Q", format=".1f"),
                ],
            )
            .properties(height=360, title="Throughput vs batch size (higher = better)")
        )
        label_layer = chart_tp + chart_tp.mark_text(
            align="center", dy=-10, fontSize=11, fontWeight="bold", color="#e2e8f0"
        ).encode(text=alt.Text("Throughput (samp/s):Q", format=".0f"))
        st.altair_chart(label_layer, use_container_width=True)
        st.caption(
            "Throughput = samples processed per second. "
            "Larger batches typically yield higher throughput due to GPU/vectorisation benefits."
        )

# ─────────────────────────────────────────────────────────────────────────────
# 4 · Stress Test
# ─────────────────────────────────────────────────────────────────────────────
if _st is not None:
    st.divider()
    st.markdown('<p class="section-title">Stress Test</p>', unsafe_allow_html=True)
    st.caption(
        f"**{_st['n_inferences']}** consecutive predict() calls · "
        f"batch size **{_st['batch_size']}** · "
        f"total duration **{_st['total_time_s']:.2f} s**"
    )

    # Key metrics
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Mean latency", _fmt_ms(_st.get("mean_latency_ms")))
    s2.metric(
        "Std",
        _fmt_ms(_st.get("std_latency_ms")),
        help="Standard deviation of per-inference latency.",
    )
    s3.metric("p50", _fmt_ms(_st.get("p50_ms")))
    s4.metric("p95", _fmt_ms(_st.get("p95_ms")))
    s5.metric("p99", _fmt_ms(_st.get("p99_ms")))
    s6.metric(
        "Throughput",
        f"{_st.get('throughput_samples_per_sec', 0):.1f} samp/s",
    )

    sr1, sr2 = st.columns(2)
    sr1.metric("Peak RAM", _fmt_mb(_st.get("peak_ram_mb")))
    if _has_gpu and _st.get("peak_gpu_mb") is not None:
        sr2.metric("Peak GPU mem", _fmt_mb(_st.get("peak_gpu_mb")))
    else:
        sr2.metric("Peak GPU mem", "N/A")

    tab_timeline, tab_dist, tab_pct = st.tabs(
        ["📈 Latency timeline", "📊 Distribution", "🎯 Percentile table"]
    )

    with tab_timeline:
        timeline = _st.get("latency_timeline", [])
        if timeline:
            # Downsample for display if very long (keep every Nth point)
            max_pts = 500
            step = max(1, len(timeline) // max_pts)
            ts_disp = timeline[::step]
            df_tl = pd.DataFrame(
                {
                    "Inference #": [i * step for i in range(len(ts_disp))],
                    "Latency (ms)": ts_disp,
                }
            )
            # Rolling mean (window = 10 % of visible points)
            w = max(3, len(ts_disp) // 10)
            df_tl["Rolling mean"] = (
                pd.Series(ts_disp).rolling(w, center=True, min_periods=1).mean().values
            )
            df_long_tl = df_tl.melt(
                id_vars="Inference #",
                value_vars=["Latency (ms)", "Rolling mean"],
                var_name="Series",
                value_name="ms",
            )
            chart_tl = (
                alt.Chart(df_long_tl)
                .mark_line(opacity=0.85)
                .encode(
                    x=alt.X("Inference #:Q", title="Inference call index"),
                    y=alt.Y("ms:Q", title="Latency (ms)"),
                    color=alt.Color(
                        "Series:N",
                        scale=alt.Scale(
                            domain=["Latency (ms)", "Rolling mean"],
                            range=["#60a5fa", "#f59e0b"],
                        ),
                    ),
                    strokeWidth=alt.condition(
                        alt.datum.Series == "Rolling mean",
                        alt.value(2.5),
                        alt.value(1.0),
                    ),
                    tooltip=[
                        "Inference #",
                        "Series",
                        alt.Tooltip("ms:Q", format=".2f"),
                    ],
                )
                .properties(
                    height=320,
                    title="Per-inference latency over time (blue=raw, amber=rolling mean)",
                )
            )
            st.altair_chart(chart_tl, use_container_width=True)
            if step > 1:
                st.caption(
                    f"Displaying every {step}th point for performance ({len(timeline)} total)."
                )
        else:
            st.info("No timeline data available.")

    with tab_dist:
        timeline = _st.get("latency_timeline", [])
        if timeline:
            df_dist = pd.DataFrame({"Latency (ms)": timeline})
            chart_dist = (
                alt.Chart(df_dist)
                .mark_bar(opacity=0.85, **_bar_props)
                .encode(
                    x=alt.X(
                        "Latency (ms):Q",
                        bin=alt.Bin(maxbins=40),
                        title="Latency (ms)",
                    ),
                    y=alt.Y("count():Q", title="Inference count"),
                    color=alt.value("#60a5fa"),
                    tooltip=[
                        alt.Tooltip("Latency (ms):Q", format=".2f", bin=True),
                        "count():Q",
                    ],
                )
                .properties(height=300, title="Latency distribution across all inferences")
            )
            # Percentile rule lines
            for pct_name, pct_val, color in [
                ("p50", _st.get("p50_ms", 0), "#34d399"),
                ("p95", _st.get("p95_ms", 0), "#f59e0b"),
                ("p99", _st.get("p99_ms", 0), "#f87171"),
            ]:
                rule = (
                    alt.Chart(pd.DataFrame({"x": [pct_val], "label": [pct_name]}))
                    .mark_rule(strokeDash=[4, 3], strokeWidth=1.5, color=color)
                    .encode(x="x:Q")
                )
                chart_dist = chart_dist + rule
            st.altair_chart(chart_dist, use_container_width=True)
            st.caption(
                "Green = p50, amber = p95, red = p99. "
                "Narrow distribution = consistent latency; long right tail = occasional spikes."
            )
        else:
            st.info("No distribution data available.")

    with tab_pct:
        pct_rows = [
            ("Min", _fmt_ms(_st.get("min_ms"))),
            ("p50 (median)", _fmt_ms(_st.get("p50_ms"))),
            ("Mean", _fmt_ms(_st.get("mean_latency_ms"))),
            ("p95", _fmt_ms(_st.get("p95_ms"))),
            ("p99", _fmt_ms(_st.get("p99_ms"))),
            ("Max", _fmt_ms(_st.get("max_ms"))),
            ("Std dev", _fmt_ms(_st.get("std_latency_ms"))),
            ("Total time (s)", f"{_st.get('total_time_s', 0):.3f} s"),
            ("Throughput", f"{_st.get('throughput_samples_per_sec', 0):.1f} samp/s"),
            ("Peak RAM", _fmt_mb(_st.get("peak_ram_mb"))),
            (
                "Peak GPU mem",
                _fmt_mb(_st.get("peak_gpu_mb")) if _has_gpu else "N/A",
            ),
        ]
        st.dataframe(
            pd.DataFrame(pct_rows, columns=["Metric", "Value"]),
            use_container_width=True,
            hide_index=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# 5 · Summary table
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Summary</p>', unsafe_allow_html=True)

summary_rows = [
    ("Model", report.model_name),
    ("Generated at", report.generated_at),
    ("Data source", report.dataset_id or "Synthetic"),
    ("TensorFlow", _sys.get("tensorflow_version", "—")),
    ("GPU", gpu_label),
    ("File size", _fmt_mb(_lp.get("file_size_mb"))),
    ("Parameters", _fmt_params(_lp.get("param_count"))),
    ("Load time", f"{_lp.get('load_time_s', 0):.3f} s"),
    ("RAM delta (load)", _fmt_mb(_lp.get("ram_delta_mb"))),
]

if _bi:
    fastest_b = min(_bi, key=lambda b: b["per_sample_ms"])
    fastest_tput = max(_bi, key=lambda b: b["throughput_samples_per_sec"])
    summary_rows += [
        (
            "Single-sample latency (bs=1)",
            _fmt_ms(next((b["mean_latency_ms"] for b in _bi if b["batch_size"] == 1), None)),
        ),
        (
            "Most efficient batch size",
            f"{fastest_b['batch_size']} ({fastest_b['per_sample_ms']:.3f} ms/sample)",
        ),
        (
            "Peak throughput",
            f"{fastest_tput['throughput_samples_per_sec']:.1f} samp/s (bs={fastest_tput['batch_size']})",
        ),
    ]

if _st:
    summary_rows += [
        ("Stress: mean latency", _fmt_ms(_st.get("mean_latency_ms"))),
        ("Stress: p99 latency", _fmt_ms(_st.get("p99_ms"))),
        ("Stress: peak RAM", _fmt_mb(_st.get("peak_ram_mb"))),
        ("Stress: throughput", f"{_st.get('throughput_samples_per_sec', 0):.1f} samp/s"),
    ]

df_summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
st.dataframe(df_summary, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6 · Export
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Export</p>', unsafe_allow_html=True)

ex1, ex2, _ = st.columns([1, 1, 3])

# CSV export of batch inference data
with ex1:
    if _bi:
        st.download_button(
            "⬇ Batch CSV",
            data=df_bi.to_csv(index=False),
            file_name=f"hw_batch_{report.model_name.replace('.keras', '')}.csv",
            mime="text/csv",
        )
    else:
        st.caption("No batch data to export.")

# Save full JSON report
with ex2:
    save_name = (
        st.text_input(
            "Report name",
            value="default",
            key="hw_save_name",
            label_visibility="collapsed",
        ).strip()
        or "default"
    )
    if st.button("💾 Save report", key="hw_save_btn"):
        try:
            p = save_report(report, name=save_name)
            st.success(f"Saved as **{p.name}**")
        except Exception as exc:
            st.error(f"Save failed: {exc}")
