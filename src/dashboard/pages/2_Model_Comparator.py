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

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import PREPROCESS_PATH
from src.dashboard.dataset_quality import get_available_datasets
from src.dashboard.model_comparator import (
    ModelComparisonResult,
    get_available_models,
    list_saved_comparisons,
    load_comparison,
    run_model_comparison,
    save_comparison,
)
from src.emg_movement.gestures import ALL_GESTURES

id_to_gesture = dict(enumerate(ALL_GESTURES))

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

/* ── Health cards ── */
.health-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(148,163,184,0.1);
    border-left: 3px solid #334155;
    border-radius: 0.65rem;
    padding: 1rem 1.15rem;
    margin-bottom: 0.6rem;
}
.health-card-healthy { border-left-color: #059669; }
.health-card-broken  { border-left-color: #dc2626; }
.health-card-nodata  { border-left-color: #2563eb; }
.health-card-failed  { border-left-color: #9333ea; }

.health-model-name {
    font-weight: 700;
    font-size: 0.92rem;
    color: #e2e8f0;
    margin-bottom: 0.45rem;
}
.health-meta {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* ── Status badges ── */
.badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.18rem 0.6rem;
    border-radius: 999px;
    letter-spacing: 0.04em;
}
.badge-healthy { background: linear-gradient(135deg,#065f46,#059669); color:#fff; }
.badge-broken  { background: linear-gradient(135deg,#7f1d1d,#b91c1c); color:#fff; }
.badge-nodata  { background: linear-gradient(135deg,#1e3a5f,#1d4ed8); color:#fff; }
.badge-failed  { background: linear-gradient(135deg,#4a044e,#9333ea); color:#fff; }
.badge-warn    { background: linear-gradient(135deg,#78350f,#d97706); color:#fff; }

.broken-reason { font-size:0.77rem; color:#f87171; margin-top:0.28rem; }
.shape-ok      { font-size:0.77rem; color:#6ee7b7; margin-top:0.28rem; }
.shape-bad     { font-size:0.77rem; color:#f87171; margin-top:0.28rem; }

/* ── Empty state ── */
.empty-state { text-align:center; padding:4rem 1rem 3rem; color:#475569; }
.empty-icon  { font-size:3.5rem; margin-bottom:0.75rem; }
.empty-title { font-size:1.15rem; font-weight:600; color:#94a3b8; }
.empty-body  { font-size:0.9rem; margin-top:0.4rem; }
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
# Sidebar — Models
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Models")
available_models = get_available_models()
if not available_models:
    st.sidebar.error("No .keras models found in src/models/")
    st.error("No models found. Train a model first.")
    st.stop()

_all_btn, _none_btn = st.sidebar.columns(2)
if _all_btn.button("All", use_container_width=True, key="mc_sel_all"):
    st.session_state["mc_default_sel"] = available_models
if _none_btn.button("None", use_container_width=True, key="mc_sel_none"):
    st.session_state["mc_default_sel"] = []

_default_sel = st.session_state.get("mc_default_sel", available_models)
selected_models: list[str] = st.sidebar.multiselect(
    "Choose models to compare",
    options=available_models,
    default=_default_sel,
    key="mc_model_multisel",
)

# ── Dataset ────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("Dataset (optional)")
st.sidebar.caption("Enable to compute accuracy, loss, and all inference metrics.")

available_datasets: list[str] = []
if os.path.isdir(PREPROCESS_PATH):
    available_datasets = get_available_datasets()

use_dataset: bool = st.sidebar.toggle("Evaluate on dataset", value=bool(available_datasets))
selected_dataset: str | None = None

if use_dataset:
    if not available_datasets:
        st.sidebar.warning("No preprocessed datasets found.")
    else:
        selected_dataset = st.sidebar.selectbox(
            "Dataset", options=available_datasets, key="mc_dataset_sel"
        )

# ── Inference settings ─────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("Inference settings")

n_windows: int = st.sidebar.slider(
    "Evaluation windows",
    min_value=50,
    max_value=2000,
    value=300,
    step=50,
    help="Windows randomly sampled (seed 42) for accuracy/loss/F1.",
    disabled=not use_dataset,
)
n_timing: int = st.sidebar.slider(
    "Timing samples",
    min_value=10,
    max_value=100,
    value=30,
    step=5,
    help="Single-window predict() calls per model for latency measurement.",
    disabled=not use_dataset,
)

# ── Saved comparisons ──────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("Saved comparisons")

saved_comps = list_saved_comparisons()
if saved_comps:
    comp_labels = [f"{name}  ({saved_at[:16]})" for name, saved_at, _ in saved_comps]
    comp_paths = [p for _, _, p in saved_comps]
    chosen_idx: int = st.sidebar.selectbox(
        "Load a saved comparison",
        options=range(len(comp_labels)),
        format_func=lambda i: comp_labels[i],
        key="mc_saved_sel",
    )
    _c_load, _c_del = st.sidebar.columns(2)
    if _c_load.button("Load", use_container_width=True, key="mc_load_btn"):
        try:
            loaded_results, loaded_cfg = load_comparison(comp_paths[chosen_idx])
            st.session_state["mc_results"] = loaded_results
            st.session_state["mc_config"] = loaded_cfg
            st.sidebar.success("Loaded.")
        except Exception as exc:
            st.sidebar.error(f"Load failed: {exc}")
    if _c_del.button("Delete", use_container_width=True, key="mc_del_btn"):
        try:
            comp_paths[chosen_idx].unlink()
            st.sidebar.success("Deleted.")
        except Exception as exc:
            st.sidebar.error(f"Delete failed: {exc}")
else:
    st.sidebar.caption("_No saved comparisons yet._")

# ── Run button ─────────────────────────────────────────────────────────────
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
    "Benchmark Keras models side-by-side — accuracy, Top-K, F1, confusion matrix, "
    "confidence & entropy, inference latency, shape validation, and architecture."
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

    cfg_new = {
        "dataset": selected_dataset,
        "n_windows": n_windows,
        "n_timing": n_timing,
        "use_dataset": use_dataset,
    }
    st.session_state["mc_results"] = results
    st.session_state["mc_config"] = cfg_new

    # Auto-save prompt in sidebar
    with st.sidebar.expander("Save this comparison", expanded=False):
        save_name = (
            st.text_input("Name", value="default", key="mc_save_name_input").strip() or "default"
        )
        if st.button("Save", key="mc_save_btn"):
            try:
                path = save_comparison(results, cfg_new, name=save_name)
                st.success(f"Saved as **{path.name}**")
            except Exception as exc:
                st.error(f"Save failed: {exc}")

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
                Select models in the sidebar and click <strong>▶ Run Comparison</strong>,
                or load a saved comparison.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Shared helpers ─────────────────────────────────────────────────────────
has_inference = any(r.accuracy is not None for r in results)

MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}


def _sort_key(r: ModelComparisonResult) -> float:
    return r.accuracy if r.accuracy is not None else -2.0


sorted_results = sorted(results, key=_sort_key, reverse=True)


def _fmt_pct(v: float | None, decimals: int = 2) -> str:
    return f"{v * 100:.{decimals}f} %" if v is not None else "—"


def _fmt_loss(v: float | None) -> str:
    if v is None:
        return "—"
    return "NaN" if math.isnan(v) else f"{v:.4f}"


def _fmt_params(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


def _fmt_lat(mean: float | None, std: float | None) -> str:
    return f"{mean:.1f} ± {std:.1f}" if mean is not None else "—"


def _short_gesture(label: int, maxlen: int = 18) -> str:
    name = id_to_gesture.get(label, f"cls_{label}")
    return name[:maxlen] + ("…" if len(name) > maxlen else "")


_bar_props = {"cornerRadiusTopLeft": 5, "cornerRadiusTopRight": 5}


def _label_layer(base: alt.Chart, field_name: str, fmt: str) -> alt.LayerChart:
    return base + base.mark_text(
        align="center", dy=-10, fontSize=11, fontWeight="bold", color="#e2e8f0"
    ).encode(text=alt.Text(f"{field_name}:Q", format=fmt))


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Overview
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Overview</p>', unsafe_allow_html=True)

n_healthy = sum(1 for r in results if r.loaded and not r.is_broken)
n_broken = sum(1 for r in results if r.is_broken)
best_acc = max((r.accuracy for r in results if r.accuracy is not None), default=None)
best_f1 = max((r.macro_f1 for r in results if r.macro_f1 is not None), default=None)
fastest = min(
    (r for r in results if r.mean_inference_ms is not None),
    key=lambda r: r.mean_inference_ms,  # type: ignore[arg-type]
    default=None,
)

ov1, ov2, ov3, ov4, ov5, ov6 = st.columns(6)
ov1.metric("Models tested", len(results))
ov2.metric("Healthy", n_healthy)
ov3.metric("Broken / failed", n_broken)
ov4.metric("Best accuracy", _fmt_pct(best_acc, 1))
ov5.metric("Best macro F1", _fmt_pct(best_f1, 1))
ov6.metric(
    "Fastest model",
    fastest.name.replace(".keras", "") if fastest else "—",
    help="Lowest mean single-window inference latency.",
)

if cfg.get("use_dataset") and cfg.get("dataset"):
    st.caption(
        f"Evaluated on dataset **{cfg['dataset']}** · "
        f"**{cfg.get('n_windows', 0)}** windows (random sample, seed 42) · "
        f"latency: **{cfg.get('n_timing', 0)}** single-window calls per model"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 2 · Ranking — two tabs: Performance | Technical
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Ranking</p>', unsafe_allow_html=True)

rank_perf_tab, rank_tech_tab = st.tabs(["📈 Performance", "⚙️ Technical"])

perf_rows = []
tech_rows = []

for rank, r in enumerate(sorted_results, start=1):
    medal = MEDALS.get(rank, str(rank))
    model_label = r.name.replace(".keras", "")

    if not r.loaded:
        status = "❌ Load failed"
    elif r.is_broken:
        status = "⚠️ Broken"
    elif r.accuracy is None:
        status = "ℹ️ Metadata only"
    else:
        status = "✅ Healthy"

    perf_rows.append(
        {
            "Rank": medal,
            "Model": model_label,
            "Status": status,
            "Accuracy": _fmt_pct(r.accuracy),
            "Top-3": _fmt_pct(r.top3_accuracy),
            "Top-5": _fmt_pct(r.top5_accuracy),
            "Macro F1": _fmt_pct(r.macro_f1),
        }
    )

    shape_str = (
        "✅ OK"
        if r.shape_compatible is True
        else ("❌ Mismatch" if r.shape_compatible is False else "—")
    )
    tech_rows.append(
        {
            "Model": model_label,
            "Loss": _fmt_loss(r.loss),
            "Size (MB)": f"{r.file_size_mb:.2f}",
            "Parameters": _fmt_params(r.param_count),
            "Latency (ms ± σ)": _fmt_lat(r.mean_inference_ms, r.std_inference_ms),
            "Shape": shape_str,
            "Modified": r.file_modified or "—",
        }
    )

_col_cfg_perf = {
    "Rank": st.column_config.TextColumn("Rank", width="small"),
    "Model": st.column_config.TextColumn("Model", width="medium"),
    "Status": st.column_config.TextColumn("Status"),
    "Accuracy": st.column_config.TextColumn("Accuracy"),
    "Top-3": st.column_config.TextColumn("Top-3"),
    "Top-5": st.column_config.TextColumn("Top-5"),
    "Macro F1": st.column_config.TextColumn("Macro F1"),
}
_col_cfg_tech = {
    "Model": st.column_config.TextColumn("Model", width="medium"),
    "Loss": st.column_config.TextColumn("Loss"),
    "Size (MB)": st.column_config.TextColumn("Size (MB)"),
    "Parameters": st.column_config.TextColumn("Parameters"),
    "Latency (ms ± σ)": st.column_config.TextColumn("Latency (ms ± σ)"),
    "Shape": st.column_config.TextColumn("Shape check"),
    "Modified": st.column_config.TextColumn("Last modified"),
}

with rank_perf_tab:
    st.dataframe(
        pd.DataFrame(perf_rows),
        use_container_width=True,
        hide_index=True,
        column_config=_col_cfg_perf,
    )

with rank_tech_tab:
    st.dataframe(
        pd.DataFrame(tech_rows),
        use_container_width=True,
        hide_index=True,
        column_config=_col_cfg_tech,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3 · Visual comparison — Accuracy | Latency | Size & Params | Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Visual Comparison</p>', unsafe_allow_html=True)

chart_rows = [
    {
        "Model": r.name.replace(".keras", ""),
        "Accuracy (%)": r.accuracy * 100 if r.accuracy is not None else None,
        "Top-3 (%)": r.top3_accuracy * 100 if r.top3_accuracy is not None else None,
        "Top-5 (%)": r.top5_accuracy * 100 if r.top5_accuracy is not None else None,
        "Macro F1 (%)": r.macro_f1 * 100 if r.macro_f1 is not None else None,
        "Inference (ms)": r.mean_inference_ms,
        "Size (MB)": r.file_size_mb,
        "Parameters (M)": r.param_count / 1e6 if r.param_count else None,
    }
    for r in results
]
df_chart = pd.DataFrame(chart_rows)

tab_acc, tab_topk, tab_lat, tab_size, tab_cm = st.tabs(
    ["📊 Accuracy & F1", "🎯 Top-K", "⚡ Latency", "💾 Size & Params", "🔲 Confusion Matrix"]
)

# ── Accuracy & F1 tab ─────────────────────────────────────────────────────
with tab_acc:
    if has_inference:
        # Fold accuracy, macro F1 into long format for grouped bars
        acc_f1_rows = []
        for r in results:
            model_label = r.name.replace(".keras", "")
            if r.accuracy is not None:
                acc_f1_rows.append(
                    {
                        "Model": model_label,
                        "Metric": "Accuracy",
                        "Value (%)": r.accuracy * 100,
                    }
                )
            if r.macro_f1 is not None:
                acc_f1_rows.append(
                    {
                        "Model": model_label,
                        "Metric": "Macro F1",
                        "Value (%)": r.macro_f1 * 100,
                    }
                )

        df_acc_f1 = pd.DataFrame(acc_f1_rows)
        chart_af = (
            alt.Chart(df_acc_f1)
            .mark_bar(**_bar_props)
            .encode(
                x=alt.X("Model:N", axis=alt.Axis(labelAngle=-15), title=None),
                y=alt.Y("Value (%):Q", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Metric:N", scale=alt.Scale(scheme="tableau10")),
                xOffset="Metric:N",
                tooltip=["Model", "Metric", alt.Tooltip("Value (%):Q", format=".2f")],
            )
            .properties(height=380)
        )
        st.altair_chart(chart_af, use_container_width=True)
        st.caption("Grouped bars — Accuracy vs. Macro F1 per model.")
    else:
        st.info("Enable **Evaluate on dataset** in the sidebar and re-run.")

# ── Top-K tab ─────────────────────────────────────────────────────────────
with tab_topk:
    if has_inference:
        topk_rows = []
        for r in results:
            ml = r.name.replace(".keras", "")
            for k_label, val in [
                ("Top-1", r.accuracy),
                ("Top-3", r.top3_accuracy),
                ("Top-5", r.top5_accuracy),
            ]:
                if val is not None:
                    topk_rows.append({"Model": ml, "K": k_label, "Accuracy (%)": val * 100})
        df_topk = pd.DataFrame(topk_rows)
        chart_topk = (
            alt.Chart(df_topk)
            .mark_bar(**_bar_props)
            .encode(
                x=alt.X("Model:N", axis=alt.Axis(labelAngle=-15), title=None),
                y=alt.Y("Accuracy (%):Q", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color(
                    "K:N",
                    scale=alt.Scale(scheme="blues"),
                    sort=["Top-1", "Top-3", "Top-5"],
                ),
                xOffset="K:N",
                tooltip=["Model", "K", alt.Tooltip("Accuracy (%):Q", format=".2f")],
            )
            .properties(height=380)
        )
        st.altair_chart(chart_topk, use_container_width=True)
        st.caption(
            "Top-K accuracy: the true gesture is in the model's top-K predicted classes. "
            "For 52 classes, Top-3 / Top-5 is especially meaningful."
        )
    else:
        st.info("Enable **Evaluate on dataset** in the sidebar and re-run.")

# ── Latency tab ───────────────────────────────────────────────────────────
with tab_lat:
    if any(r.mean_inference_ms is not None for r in results):
        df_lat = df_chart.dropna(subset=["Inference (ms)"])
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
            alt.Chart(df_lat)
            .mark_bar(**_bar_props)
            .encode(
                x=alt.X("Model:N", sort="y", axis=alt.Axis(labelAngle=-15), title=None),
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
            _label_layer(base_t, "Inference (ms)", ".1f") + err_chart,
            use_container_width=True,
        )
        st.caption(
            "Error bars = ±1 σ across timing samples. "
            "Includes TensorFlow `predict()` API overhead, not raw GPU time."
        )
    else:
        st.info("Enable **Evaluate on dataset** in the sidebar and re-run.")

# ── Size & Params tab ─────────────────────────────────────────────────────
with tab_size:
    sp1, sp2 = st.columns(2)
    with sp1:
        base_s = (
            alt.Chart(df_chart)
            .mark_bar(**_bar_props)
            .encode(
                x=alt.X("Model:N", sort="y", axis=alt.Axis(labelAngle=-15), title=None),
                y=alt.Y("Size (MB):Q", title="File size (MB)"),
                color=alt.Color("Model:N", legend=None),
                tooltip=["Model", alt.Tooltip("Size (MB):Q", format=".2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(_label_layer(base_s, "Size (MB)", ".2f"), use_container_width=True)
    with sp2:
        df_params = df_chart.dropna(subset=["Parameters (M)"])
        if not df_params.empty:
            base_p = (
                alt.Chart(df_params)
                .mark_bar(**_bar_props)
                .encode(
                    x=alt.X("Model:N", sort="-y", axis=alt.Axis(labelAngle=-15), title=None),
                    y=alt.Y("Parameters (M):Q", title="Parameters (M)"),
                    color=alt.Color("Model:N", legend=None),
                    tooltip=["Model", alt.Tooltip("Parameters (M):Q", format=".3f")],
                )
                .properties(height=320)
            )
            st.altair_chart(_label_layer(base_p, "Parameters (M)", ".3f"), use_container_width=True)
    st.caption("Smaller models are preferable for on-device / real-time deployment.")

# ── Confusion Matrix tab ──────────────────────────────────────────────────
with tab_cm:
    models_with_cm = [r for r in sorted_results if r.confusion_matrix is not None]
    if not models_with_cm:
        st.info("Enable **Evaluate on dataset** in the sidebar and re-run.")
    else:
        st.caption(
            "Rows = true gesture, columns = predicted gesture. "
            "Diagonal = correct predictions. Darker = more samples."
        )
        for r in models_with_cm:
            st.markdown(f"**{r.name.replace('.keras', '')}**")
            labels = r.class_labels or []
            cm_arr = np.array(r.confusion_matrix)
            # Build long-format DataFrame
            cm_long = []
            for i, true_lbl in enumerate(labels):
                for j, pred_lbl in enumerate(labels):
                    cm_long.append(
                        {
                            "True": _short_gesture(true_lbl),
                            "Predicted": _short_gesture(pred_lbl),
                            "True (full)": id_to_gesture.get(true_lbl, str(true_lbl)),
                            "Predicted (full)": id_to_gesture.get(pred_lbl, str(pred_lbl)),
                            "Count": int(cm_arr[i][j]),
                        }
                    )
            df_cm = pd.DataFrame(cm_long)
            n_cls = len(labels)
            h = max(320, min(800, n_cls * 14))
            chart_cm = (
                alt.Chart(df_cm)
                .mark_rect()
                .encode(
                    x=alt.X(
                        "Predicted:N",
                        sort=[_short_gesture(lbl) for lbl in labels],
                        axis=alt.Axis(labelAngle=90, labelFontSize=9, labelLimit=140),
                        title="Predicted",
                    ),
                    y=alt.Y(
                        "True:N",
                        sort=[_short_gesture(lbl) for lbl in labels],
                        axis=alt.Axis(labelFontSize=9, labelLimit=140),
                        title="True",
                    ),
                    color=alt.Color(
                        "Count:Q",
                        scale=alt.Scale(scheme="blues"),
                        legend=alt.Legend(title="Count"),
                    ),
                    tooltip=[
                        alt.Tooltip("True (full):N", title="True gesture"),
                        alt.Tooltip("Predicted (full):N", title="Predicted gesture"),
                        "Count:Q",
                    ],
                )
                .properties(height=h)
            )
            st.altair_chart(chart_cm, use_container_width=True)

        # Diff heatmap when exactly 2 models with matching labels
        if (
            len(models_with_cm) == 2
            and models_with_cm[0].class_labels == models_with_cm[1].class_labels
        ):
            st.divider()
            m_a, m_b = models_with_cm[0], models_with_cm[1]
            st.markdown(
                f"**Δ Confusion matrix** — "
                f"{m_a.name.replace('.keras', '')} minus "
                f"{m_b.name.replace('.keras', '')}"
            )
            labels = m_a.class_labels or []
            cm_diff = np.array(m_a.confusion_matrix) - np.array(m_b.confusion_matrix)
            max_abs = int(np.abs(cm_diff).max()) or 1
            diff_long = []
            for i, tl in enumerate(labels):
                for j, pl in enumerate(labels):
                    diff_long.append(
                        {
                            "True": _short_gesture(tl),
                            "Predicted": _short_gesture(pl),
                            "True (full)": id_to_gesture.get(tl, str(tl)),
                            "Predicted (full)": id_to_gesture.get(pl, str(pl)),
                            "Δ Count": int(cm_diff[i][j]),
                        }
                    )
            df_diff = pd.DataFrame(diff_long)
            n_cls = len(labels)
            h_diff = max(320, min(800, n_cls * 14))
            chart_diff = (
                alt.Chart(df_diff)
                .mark_rect()
                .encode(
                    x=alt.X(
                        "Predicted:N",
                        sort=[_short_gesture(lbl) for lbl in labels],
                        axis=alt.Axis(labelAngle=90, labelFontSize=9, labelLimit=140),
                        title="Predicted",
                    ),
                    y=alt.Y(
                        "True:N",
                        sort=[_short_gesture(lbl) for lbl in labels],
                        axis=alt.Axis(labelFontSize=9, labelLimit=140),
                        title="True",
                    ),
                    color=alt.Color(
                        "Δ Count:Q",
                        scale=alt.Scale(
                            scheme="blueorange",
                            domain=[-max_abs, 0, max_abs],
                        ),
                        legend=alt.Legend(title="Δ Count"),
                    ),
                    tooltip=[
                        alt.Tooltip("True (full):N", title="True gesture"),
                        alt.Tooltip("Predicted (full):N", title="Predicted gesture"),
                        "Δ Count:Q",
                    ],
                )
                .properties(height=h_diff)
            )
            st.altair_chart(chart_diff, use_container_width=True)
            st.caption(
                f"Blue = {m_a.name.replace('.keras', '')} predicts more in that cell. "
                f"Orange = {m_b.name.replace('.keras', '')} predicts more. "
                "On the diagonal: blue = model A more correct for that gesture."
            )

# ─────────────────────────────────────────────────────────────────────────────
# 4 · Health monitor
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Health Monitor</p>', unsafe_allow_html=True)

n_cols = min(len(sorted_results), 3)
h_cols = st.columns(n_cols)

for i, r in enumerate(sorted_results):
    col = h_cols[i % n_cols]
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

        card = (
            f'<div class="health-card {card_cls}">'
            f'<div class="health-model-name">{html.escape(r.name.replace(".keras", ""))}</div>'
            f'<span class="badge {badge_cls}">{badge_txt}</span>'
        )

        if r.broken_reasons:
            for reason in r.broken_reasons:
                card += f'<div class="broken-reason">• {html.escape(reason)}</div>'
        elif not r.loaded and r.load_error:
            card += f'<div class="broken-reason">{html.escape(r.load_error[:120])}</div>'
        elif r.accuracy is not None:
            card += (
                f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:0.45rem;">'
                f'Accuracy <strong style="color:#e2e8f0">{r.accuracy * 100:.1f}%</strong>'
                f" &nbsp;·&nbsp; "
                f'F1 <strong style="color:#e2e8f0">{r.macro_f1 * 100:.1f}%</strong>'
                if r.macro_f1 is not None
                else f'Accuracy <strong style="color:#e2e8f0">{r.accuracy * 100:.1f}%</strong>'
                f"</div>"
            )
            card += "</div>"  # close the inner div

        # Shape info
        if r.model_input_shape is not None and r.dataset_input_shape is not None:
            shape_cls = "shape-ok" if r.shape_compatible else "shape-bad"
            shape_icon = "✓" if r.shape_compatible else "✗"
            card += (
                f'<div class="{shape_cls}">'
                f"{shape_icon} model {r.model_input_shape} vs data {r.dataset_input_shape}"
                f"</div>"
            )

        # File modified
        if r.file_modified:
            card += f'<div class="health-meta">📅 {html.escape(r.file_modified)}</div>'

        card += "</div>"
        st.markdown(card, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5 · Per-class metrics (accuracy + precision + recall + F1)
# ─────────────────────────────────────────────────────────────────────────────
models_with_pc = [r for r in sorted_results if r.per_class_accuracy]

if models_with_pc:
    st.divider()
    st.markdown('<p class="section-title">Per-class Metrics</p>', unsafe_allow_html=True)

    for r in models_with_pc:
        macro_f1_str = f"  ·  Macro F1: {r.macro_f1 * 100:.1f}%" if r.macro_f1 is not None else ""
        with st.expander(
            f"📋 {r.name.replace('.keras', '')} — per-gesture breakdown{macro_f1_str}",
            expanded=False,
        ):
            pc_rows = []
            for lbl in sorted(r.per_class_accuracy.keys()):
                pc_rows.append(
                    {
                        "Label": lbl,
                        "Gesture": id_to_gesture.get(lbl, f"cls_{lbl}"),
                        "Accuracy": round(r.per_class_accuracy.get(lbl, 0.0), 4),
                        "Precision": round((r.per_class_precision or {}).get(lbl, 0.0), 4),
                        "Recall": round((r.per_class_recall or {}).get(lbl, 0.0), 4),
                        "F1": round((r.per_class_f1 or {}).get(lbl, 0.0), 4),
                    }
                )
            df_pc = pd.DataFrame(pc_rows)

            # F1 heatmap bar chart
            chart_f1 = (
                alt.Chart(df_pc)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X(
                        "Gesture:N",
                        sort=alt.EncodingSortField("F1", order="descending"),
                        axis=alt.Axis(labelAngle=90, labelLimit=200, labelFontSize=9),
                        title=None,
                    ),
                    y=alt.Y("F1:Q", scale=alt.Scale(domain=[0, 1]), title="F1 Score"),
                    color=alt.Color(
                        "F1:Q",
                        scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
                        legend=None,
                    ),
                    tooltip=[
                        "Gesture",
                        alt.Tooltip("Accuracy:Q", format=".3f"),
                        alt.Tooltip("Precision:Q", format=".3f"),
                        alt.Tooltip("Recall:Q", format=".3f"),
                        alt.Tooltip("F1:Q", format=".3f"),
                    ],
                )
                .properties(height=280, title="F1 Score per gesture (sorted, green=high)")
            )
            st.altair_chart(chart_f1, use_container_width=True)

            # Combined metrics table with progress columns
            st.dataframe(
                df_pc[["Gesture", "Accuracy", "Precision", "Recall", "F1"]].sort_values(
                    "F1", ascending=False
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Gesture": st.column_config.TextColumn("Gesture"),
                    "Accuracy": st.column_config.ProgressColumn(
                        "Accuracy", min_value=0, max_value=1, format="%.3f"
                    ),
                    "Precision": st.column_config.ProgressColumn(
                        "Precision", min_value=0, max_value=1, format="%.3f"
                    ),
                    "Recall": st.column_config.ProgressColumn(
                        "Recall", min_value=0, max_value=1, format="%.3f"
                    ),
                    "F1": st.column_config.ProgressColumn(
                        "F1 Score", min_value=0, max_value=1, format="%.3f"
                    ),
                },
                height=350,
            )

# ─────────────────────────────────────────────────────────────────────────────
# 6 · Confidence & Entropy
# ─────────────────────────────────────────────────────────────────────────────
models_with_conf = [r for r in sorted_results if r.confidence_distribution]

if models_with_conf:
    st.divider()
    st.markdown('<p class="section-title">Confidence & Entropy</p>', unsafe_allow_html=True)
    st.caption(
        "**Confidence** = max(softmax) per prediction (higher = more decisive). "
        "**Entropy** = uncertainty of the full probability distribution "
        f"(max for 52 classes ≈ {math.log(52):.2f} nats; lower = more confident)."
    )

    # Summary metrics row
    conf_cols = st.columns(len(models_with_conf))
    for col, r in zip(conf_cols, models_with_conf, strict=False):
        model_lbl = r.name.replace(".keras", "")
        col.metric(
            model_lbl,
            f"conf {r.mean_confidence * 100:.1f}%" if r.mean_confidence is not None else "—",
            delta=f"entropy {r.mean_entropy:.3f}" if r.mean_entropy is not None else None,
            delta_color="inverse",
            help="Mean max-softmax confidence · Mean entropy (lower entropy = more confident)",
        )

    # Confidence distribution — faceted histograms
    conf_long = []
    for r in models_with_conf:
        for c in r.confidence_distribution:
            conf_long.append({"Model": r.name.replace(".keras", ""), "Confidence": c})
    df_conf = pd.DataFrame(conf_long)

    chart_conf = (
        alt.Chart(df_conf)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X(
                "Confidence:Q",
                bin=alt.Bin(maxbins=25),
                title="Max-softmax confidence",
            ),
            y=alt.Y("count():Q", title="Predictions"),
            color=alt.Color("Model:N", legend=None),
        )
        .properties(height=180, width=300)
        .facet(facet="Model:N", columns=min(len(models_with_conf), 3))
        .resolve_scale(y="independent")
    )
    st.altair_chart(chart_conf, use_container_width=True)
    st.caption(
        "A spike near 1.0 indicates high confidence (possibly overconfident). "
        "A spread toward 0.0 indicates many uncertain predictions."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 7 · Architecture summary
# ─────────────────────────────────────────────────────────────────────────────
models_with_arch = [r for r in sorted_results if r.layer_summary]

if models_with_arch:
    st.divider()
    st.markdown('<p class="section-title">Architecture</p>', unsafe_allow_html=True)

    for r in models_with_arch:
        n_trainable = sum(lyr["params"] for lyr in r.layer_summary if lyr.get("trainable", True))
        n_frozen = (r.param_count or 0) - n_trainable
        with st.expander(
            f"🏗️ {r.name.replace('.keras', '')} — {len(r.layer_summary)} layers · "
            f"{_fmt_params(r.param_count)} total params",
            expanded=False,
        ):
            a1, a2, a3 = st.columns(3)
            a1.metric("Total params", _fmt_params(r.param_count))
            a2.metric("Trainable", _fmt_params(n_trainable))
            a3.metric("Frozen", _fmt_params(n_frozen) if n_frozen > 0 else "0")

            df_arch = pd.DataFrame(r.layer_summary)
            st.dataframe(
                df_arch[["name", "type", "output_shape", "params", "trainable"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("Layer"),
                    "type": st.column_config.TextColumn("Type"),
                    "output_shape": st.column_config.TextColumn("Output shape"),
                    "params": st.column_config.NumberColumn("Params", format="%d"),
                    "trainable": st.column_config.CheckboxColumn("Trainable"),
                },
            )

# ─────────────────────────────────────────────────────────────────────────────
# 8 · Export
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Export</p>', unsafe_allow_html=True)

export_rows = [
    {
        "model": r.name,
        "loaded": r.loaded,
        "file_size_mb": round(r.file_size_mb, 4),
        "file_modified": r.file_modified,
        "param_count": r.param_count,
        "model_input_shape": str(r.model_input_shape) if r.model_input_shape else None,
        "dataset_input_shape": str(r.dataset_input_shape) if r.dataset_input_shape else None,
        "shape_compatible": r.shape_compatible,
        "accuracy": round(r.accuracy, 6) if r.accuracy is not None else None,
        "top3_accuracy": round(r.top3_accuracy, 6) if r.top3_accuracy is not None else None,
        "top5_accuracy": round(r.top5_accuracy, 6) if r.top5_accuracy is not None else None,
        "loss": round(r.loss, 6) if r.loss is not None and not math.isnan(r.loss) else r.loss,
        "macro_f1": round(r.macro_f1, 6) if r.macro_f1 is not None else None,
        "mean_confidence": round(r.mean_confidence, 6) if r.mean_confidence is not None else None,
        "mean_entropy": round(r.mean_entropy, 6) if r.mean_entropy is not None else None,
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

ex1, ex2, _ = st.columns([1, 1, 3])
with ex1:
    st.download_button(
        "⬇ Summary CSV",
        data=df_export.to_csv(index=False),
        file_name="model_comparison.csv",
        mime="text/csv",
    )
with ex2:
    save_name_export = (
        st.text_input(
            "Save name",
            value="default",
            key="mc_export_name",
            label_visibility="collapsed",
        ).strip()
        or "default"
    )
    if st.button("💾 Save to disk", key="mc_save_export"):
        try:
            p = save_comparison(results, cfg, name=save_name_export)
            st.success(f"Saved as **{p.name}**")
        except Exception as exc:
            st.error(f"Save failed: {exc}")
