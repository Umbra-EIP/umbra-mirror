# Model Comparator — Streamlit page

from __future__ import annotations

import html
import math
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from src.config import PREPROCESS_PATH
from src.dashboard.dataset_quality import get_available_datasets
from src.dashboard.model_comparator import (
    ModelComparisonResult,
    get_available_models,
    run_model_comparison,
)
from src.emg_movement.gestures import ALL_GESTURES

id_to_gesture = {i: g for i, g in enumerate(ALL_GESTURES)}

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Comparator | Umbra",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Layout ── */
.block-container { padding-top: 1.5rem; }

/* ── Section title ── */
.section-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.25rem;
    margin-bottom: 0.75rem;
}

/* ── Health cards ── */
.health-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-left: 3px solid #334155;
    border-radius: 0.65rem;
    padding: 1rem 1.15rem;
    margin-bottom: 0.6rem;
}
.health-card-healthy  { border-left-color: #059669; }
.health-card-broken   { border-left-color: #dc2626; }
.health-card-nodata   { border-left-color: #2563eb; }
.health-card-failed   { border-left-color: #9333ea; }

.health-model-name {
    font-weight: 700;
    font-size: 0.92rem;
    color: #e2e8f0;
    margin-bottom: 0.45rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Status badges ── */
.badge {
    display: inline-block;
    font-size: 0.73rem;
    font-weight: 700;
    padding: 0.18rem 0.6rem;
    border-radius: 999px;
    letter-spacing: 0.04em;
}
.badge-healthy { background: linear-gradient(135deg, #065f46, #059669); color: #fff; }
.badge-broken  { background: linear-gradient(135deg, #7f1d1d, #b91c1c); color: #fff; }
.badge-nodata  { background: linear-gradient(135deg, #1e3a5f, #1d4ed8); color: #fff; }
.badge-failed  { background: linear-gradient(135deg, #4a044e, #9333ea); color: #fff; }

.broken-reason {
    font-size: 0.77rem;
    color: #f87171;
    margin-top: 0.28rem;
}

/* ── Summary banner ── */
.summary-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 0.75rem;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.5rem;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 1rem 3rem;
    color: #475569;
}
.empty-icon  { font-size: 3.5rem; margin-bottom: 0.75rem; }
.empty-title { font-size: 1.15rem; font-weight: 600; color: #94a3b8; }
.empty-body  { font-size: 0.9rem; margin-top: 0.4rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "mc_results" not in st.session_state:
    st.session_state["mc_results"] = None
if "mc_config" not in st.session_state:
    st.session_state["mc_config"] = {}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Models")

available_models = get_available_models()
if not available_models:
    st.sidebar.error("No .keras models found in src/models/")
    st.error(
        "No models found. Train a model first (`python -m src.emg_movement.train …`)."
    )
    st.stop()

# All / None shortcuts
_btn_all, _btn_none = st.sidebar.columns(2)
if _btn_all.button("All", use_container_width=True, key="mc_sel_all"):
    st.session_state["mc_default_sel"] = available_models
if _btn_none.button("None", use_container_width=True, key="mc_sel_none"):
    st.session_state["mc_default_sel"] = []

_default_sel = st.session_state.get("mc_default_sel", available_models)
selected_models: list[str] = st.sidebar.multiselect(
    "Choose models to compare",
    options=available_models,
    default=_default_sel,
    key="mc_model_multisel",
)

st.sidebar.divider()
st.sidebar.header("Dataset (optional)")
st.sidebar.caption("Enable to evaluate accuracy, loss, and inference latency.")

available_datasets: list[str] = []
if os.path.isdir(PREPROCESS_PATH):
    available_datasets = get_available_datasets()

use_dataset: bool = st.sidebar.toggle(
    "Evaluate on dataset", value=bool(available_datasets)
)
selected_dataset: str | None = None

if use_dataset:
    if not available_datasets:
        st.sidebar.warning("No preprocessed datasets found — run preprocessing first.")
    else:
        selected_dataset = st.sidebar.selectbox(
            "Dataset", options=available_datasets, key="mc_dataset_sel"
        )

st.sidebar.divider()
st.sidebar.header("Inference settings")

n_windows: int = st.sidebar.slider(
    "Evaluation windows",
    min_value=50,
    max_value=2000,
    value=300,
    step=50,
    help="Windows randomly sampled (seed 42) for accuracy / loss. More windows = longer run time.",
    disabled=not use_dataset,
)
n_timing: int = st.sidebar.slider(
    "Timing samples",
    min_value=10,
    max_value=100,
    value=30,
    step=5,
    help="Number of single-window predict() calls per model for latency measurement.",
    disabled=not use_dataset,
)

st.sidebar.divider()
run_btn = st.sidebar.button(
    "▶  Run Comparison",
    type="primary",
    use_container_width=True,
    disabled=len(selected_models) == 0,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔬 Model Comparator")
st.caption(
    "Load and benchmark Keras models side-by-side — accuracy, loss, size, "
    "parameter count, inference latency, and automatic health diagnostics."
)

# ─────────────────────────────────────────────────────────────────────────────
# Run comparison
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    _status = st.empty()
    _bar = st.progress(0)

    def _cb(label: str, done: int, total: int) -> None:
        _bar.progress(done / max(total, 1))
        _status.caption(f"⏳ {label}")

    with st.spinner(""):
        results = run_model_comparison(
            model_names=selected_models,
            dataset_id=selected_dataset if use_dataset else None,
            n_windows=n_windows,
            n_timing_samples=n_timing,
            progress_callback=_cb,
        )

    _bar.empty()
    _status.empty()

    st.session_state["mc_results"] = results
    st.session_state["mc_config"] = {
        "dataset": selected_dataset,
        "n_windows": n_windows,
        "n_timing": n_timing,
        "use_dataset": use_dataset,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
results: list[ModelComparisonResult] | None = st.session_state.get("mc_results")
cfg: dict = st.session_state.get("mc_config", {})

if results is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">🔬</div>
            <div class="empty-title">No comparison run yet</div>
            <div class="empty-body">
                Select one or more models in the sidebar and click
                <strong>▶ Run Comparison</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Helpers ──────────────────────────────────────────────────────────────────

has_inference = any(r.accuracy is not None for r in results)


def _sort_key(r: ModelComparisonResult) -> float:
    if r.accuracy is not None:
        return r.accuracy
    return -2.0  # metadata-only and broken go last when ranking by accuracy


sorted_results = sorted(results, key=_sort_key, reverse=True)

MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}


def _fmt_params(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


def _fmt_loss(v: float | None) -> str:
    if v is None:
        return "—"
    if math.isnan(v):
        return "NaN"
    return f"{v:.4f}"


def _fmt_acc(v: float | None) -> str:
    return f"{v * 100:.2f} %" if v is not None else "—"


def _fmt_lat(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "—"
    return f"{mean:.1f} ± {std:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Overview summary
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Overview</p>', unsafe_allow_html=True)

n_tested = len(results)
n_healthy = sum(1 for r in results if r.loaded and not r.is_broken)
n_broken = sum(1 for r in results if r.is_broken)
best_acc = max((r.accuracy for r in results if r.accuracy is not None), default=None)
fastest = min(
    (r for r in results if r.mean_inference_ms is not None),
    key=lambda r: r.mean_inference_ms,  # type: ignore[arg-type]
    default=None,
)

ov1, ov2, ov3, ov4, ov5 = st.columns(5)
ov1.metric("Models tested", n_tested)
ov2.metric("Healthy", n_healthy)
ov3.metric("Broken / failed", n_broken)
ov4.metric(
    "Best accuracy",
    f"{best_acc * 100:.1f} %" if best_acc is not None else "—",
)
ov5.metric(
    "Fastest model",
    fastest.name.replace(".keras", "") if fastest else "—",
    help="Model with the lowest mean single-window inference latency.",
)

if cfg.get("use_dataset") and cfg.get("dataset"):
    st.caption(
        f"Evaluated on dataset **{cfg['dataset']}** · "
        f"**{cfg.get('n_windows', 0)}** windows (random sample, seed 42) · "
        f"latency averaged over **{cfg.get('n_timing', 0)}** single-window calls"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 2 · Ranking table
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Ranking</p>', unsafe_allow_html=True)

table_rows = []
for rank, r in enumerate(sorted_results, start=1):
    medal = MEDALS.get(rank, str(rank))
    if not r.loaded:
        status = "❌ Load failed"
    elif r.is_broken:
        status = "⚠️ Broken"
    elif r.accuracy is None:
        status = "ℹ️ Metadata only"
    else:
        status = "✅ Healthy"

    table_rows.append(
        {
            "Rank": medal,
            "Model": r.name.replace(".keras", ""),
            "Status": status,
            "Accuracy": _fmt_acc(r.accuracy),
            "Loss": _fmt_loss(r.loss),
            "Size (MB)": f"{r.file_size_mb:.2f}",
            "Parameters": _fmt_params(r.param_count),
            "Latency (ms ± σ)": _fmt_lat(r.mean_inference_ms, r.std_inference_ms),
        }
    )

df_table = pd.DataFrame(table_rows)
st.dataframe(
    df_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.TextColumn("Rank", width="small"),
        "Model": st.column_config.TextColumn("Model", width="medium"),
        "Status": st.column_config.TextColumn("Status"),
        "Accuracy": st.column_config.TextColumn("Accuracy"),
        "Loss": st.column_config.TextColumn("Loss"),
        "Size (MB)": st.column_config.TextColumn("Size (MB)"),
        "Parameters": st.column_config.TextColumn("Parameters"),
        "Latency (ms ± σ)": st.column_config.TextColumn("Latency (ms ± σ)"),
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# 3 · Visual comparison charts
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Visual Comparison</p>', unsafe_allow_html=True)

chart_rows = [
    {
        "Model": r.name.replace(".keras", ""),
        "Accuracy (%)": r.accuracy * 100 if r.accuracy is not None else None,
        "Inference (ms)": r.mean_inference_ms,
        "Size (MB)": r.file_size_mb,
        "Parameters (M)": r.param_count / 1e6 if r.param_count else None,
    }
    for r in results
]
df_chart = pd.DataFrame(chart_rows)

tab_acc, tab_time, tab_size, tab_params = st.tabs(
    ["📊 Accuracy", "⚡ Latency", "💾 Model Size", "🧠 Parameters"]
)

_bar_props = dict(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)


def _label_chart(base: alt.Chart, field: str, fmt: str) -> alt.LayerChart:
    return base + base.mark_text(
        align="center", dy=-10, fontSize=12, fontWeight="bold", color="#e2e8f0"
    ).encode(text=alt.Text(f"{field}:Q", format=fmt))


with tab_acc:
    if has_inference:
        df_acc = df_chart.dropna(subset=["Accuracy (%)"])
        base = (
            alt.Chart(df_acc)
            .mark_bar(**_bar_props)
            .encode(
                x=alt.X("Model:N", sort="-y", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(
                    "Accuracy (%):Q",
                    scale=alt.Scale(domain=[0, 100]),
                    title="Accuracy (%)",
                ),
                color=alt.Color("Model:N", legend=None),
                tooltip=["Model", alt.Tooltip("Accuracy (%):Q", format=".2f")],
            )
            .properties(height=380)
        )
        st.altair_chart(
            _label_chart(base, "Accuracy (%)", ".1f"), use_container_width=True
        )
    else:
        st.info(
            "Enable **Evaluate on dataset** in the sidebar and re-run to see accuracy."
        )

with tab_time:
    if any(r.mean_inference_ms is not None for r in results):
        df_time = df_chart.dropna(subset=["Inference (ms)"])

        # Build error-bar data from raw results
        err_rows = [
            {
                "Model": r.name.replace(".keras", ""),
                "mean": r.mean_inference_ms,
                "lo": max(0.0, r.mean_inference_ms - r.std_inference_ms),  # type: ignore
                "hi": r.mean_inference_ms + r.std_inference_ms,  # type: ignore
            }
            for r in results
            if r.mean_inference_ms is not None
        ]
        df_err = pd.DataFrame(err_rows)

        base_t = (
            alt.Chart(df_time)
            .mark_bar(**_bar_props)
            .encode(
                x=alt.X("Model:N", sort="y", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Inference (ms):Q", title="Mean latency (ms)"),
                color=alt.Color("Model:N", legend=None),
                tooltip=["Model", alt.Tooltip("Inference (ms):Q", format=".1f")],
            )
            .properties(height=380)
        )
        err_chart = (
            alt.Chart(df_err)
            .mark_errorbar()
            .encode(
                x=alt.X("Model:N", sort="y"),
                y=alt.Y("lo:Q", title=""),
                y2=alt.Y2("hi:Q"),
                color=alt.value("#94a3b8"),
            )
        )
        st.altair_chart(
            _label_chart(base_t, "Inference (ms)", ".1f") + err_chart,
            use_container_width=True,
        )
        st.caption(
            "Error bars show ±1 σ across timing samples. "
            "Latency includes TensorFlow `predict()` API overhead (not raw GPU time)."
        )
    else:
        st.info(
            "Enable **Evaluate on dataset** in the sidebar and re-run to see latency."
        )

with tab_size:
    base_s = (
        alt.Chart(df_chart)
        .mark_bar(**_bar_props)
        .encode(
            x=alt.X("Model:N", sort="y", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Size (MB):Q", title="File size (MB)"),
            color=alt.Color("Model:N", legend=None),
            tooltip=["Model", alt.Tooltip("Size (MB):Q", format=".2f")],
        )
        .properties(height=380)
    )
    st.altair_chart(_label_chart(base_s, "Size (MB)", ".2f"), use_container_width=True)
    st.caption("Smaller models can be preferable for on-device / edge deployment.")

with tab_params:
    df_params = df_chart.dropna(subset=["Parameters (M)"])
    if not df_params.empty:
        base_p = (
            alt.Chart(df_params)
            .mark_bar(**_bar_props)
            .encode(
                x=alt.X("Model:N", sort="-y", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Parameters (M):Q", title="Parameters (M)"),
                color=alt.Color("Model:N", legend=None),
                tooltip=["Model", alt.Tooltip("Parameters (M):Q", format=".3f")],
            )
            .properties(height=380)
        )
        st.altair_chart(
            _label_chart(base_p, "Parameters (M)", ".3f"), use_container_width=True
        )
    else:
        st.info("No parameter count available.")

# ─────────────────────────────────────────────────────────────────────────────
# 4 · Health monitor
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Health Monitor</p>', unsafe_allow_html=True)

n_cols = min(len(sorted_results), 3)
health_cols = st.columns(n_cols)

for i, r in enumerate(sorted_results):
    col = health_cols[i % n_cols]
    with col:
        if not r.loaded:
            card_cls, badge_cls, badge_txt = (
                "health-card-failed",
                "badge-failed",
                "❌ LOAD FAILED",
            )
        elif r.is_broken:
            card_cls, badge_cls, badge_txt = (
                "health-card-broken",
                "badge-broken",
                "⚠️ BROKEN",
            )
        elif r.accuracy is None:
            card_cls, badge_cls, badge_txt = (
                "health-card-nodata",
                "badge-nodata",
                "ℹ️ NO DATA",
            )
        else:
            card_cls, badge_cls, badge_txt = (
                "health-card-healthy",
                "badge-healthy",
                "✅ HEALTHY",
            )

        card_html = (
            f'<div class="health-card {card_cls}">'
            f'<div class="health-model-name">{html.escape(r.name.replace(".keras", ""))}</div>'
            f'<span class="badge {badge_cls}">{badge_txt}</span>'
        )

        if r.broken_reasons:
            for reason in r.broken_reasons:
                card_html += f'<div class="broken-reason">• {html.escape(reason)}</div>'
        elif not r.loaded and r.load_error:
            card_html += (
                f'<div class="broken-reason">{html.escape(r.load_error[:120])}</div>'
            )
        elif r.accuracy is not None:
            card_html += (
                f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:0.45rem;">'
                f'Accuracy: <strong style="color:#e2e8f0">{r.accuracy*100:.1f} %</strong>'
                f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                f'Loss: <strong style="color:#e2e8f0">{_fmt_loss(r.loss)}</strong>'
                f"</div>"
            )

        card_html += "</div>"
        st.markdown(card_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5 · Per-class accuracy (expandable per model)
# ─────────────────────────────────────────────────────────────────────────────
models_with_pc = [r for r in sorted_results if r.per_class_accuracy]

if models_with_pc:
    st.divider()
    st.markdown(
        '<p class="section-title">Per-class Accuracy</p>', unsafe_allow_html=True
    )

    for r in models_with_pc:
        with st.expander(
            f"📋 {r.name.replace('.keras', '')} — per-gesture breakdown", expanded=False
        ):
            pc_rows = [
                {
                    "Label": label,
                    "Gesture": id_to_gesture.get(label, f"class_{label}"),
                    "Accuracy (%)": round(acc * 100, 2),
                }
                for label, acc in sorted(r.per_class_accuracy.items())
            ]
            df_pc = pd.DataFrame(pc_rows)

            # Heatmap-style bar chart coloured by accuracy
            chart_pc = (
                alt.Chart(df_pc)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X(
                        "Gesture:N",
                        sort=alt.EncodingSortField("Accuracy (%)", order="descending"),
                        axis=alt.Axis(labelAngle=90, labelLimit=220, labelFontSize=10),
                        title=None,
                    ),
                    y=alt.Y(
                        "Accuracy (%):Q",
                        scale=alt.Scale(domain=[0, 100]),
                        title="Accuracy (%)",
                    ),
                    color=alt.Color(
                        "Accuracy (%):Q",
                        scale=alt.Scale(scheme="redyellowgreen", domain=[0, 100]),
                        legend=alt.Legend(title="Accuracy (%)"),
                    ),
                    tooltip=[
                        "Gesture",
                        "Label",
                        alt.Tooltip("Accuracy (%):Q", format=".1f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_pc, use_container_width=True)

            col_top, col_bot = st.columns(2)
            with col_top:
                st.markdown("**Top 5 gestures**")
                st.dataframe(
                    df_pc.sort_values("Accuracy (%)", ascending=False).head(5)[
                        ["Gesture", "Accuracy (%)"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            with col_bot:
                st.markdown("**Bottom 5 gestures**")
                st.dataframe(
                    df_pc.sort_values("Accuracy (%)").head(5)[
                        ["Gesture", "Accuracy (%)"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# 6 · Export
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Export</p>', unsafe_allow_html=True)

export_rows = [
    {
        "model": r.name,
        "loaded": r.loaded,
        "file_size_mb": round(r.file_size_mb, 4),
        "param_count": r.param_count,
        "accuracy": round(r.accuracy, 6) if r.accuracy is not None else None,
        "loss": round(r.loss, 6)
        if r.loss is not None and not math.isnan(r.loss)
        else r.loss,
        "mean_inference_ms": round(r.mean_inference_ms, 3)
        if r.mean_inference_ms is not None
        else None,
        "std_inference_ms": round(r.std_inference_ms, 3)
        if r.std_inference_ms is not None
        else None,
        "n_windows_evaluated": r.n_windows_evaluated,
        "is_broken": r.is_broken,
        "broken_reasons": "; ".join(r.broken_reasons),
    }
    for r in results
]
df_export = pd.DataFrame(export_rows)

ex1, ex2 = st.columns([1, 3])
with ex1:
    st.download_button(
        "⬇ Download comparison (CSV)",
        data=df_export.to_csv(index=False),
        file_name="model_comparison.csv",
        mime="text/csv",
    )
