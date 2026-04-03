# EEG–EMG hardware impact — Streamlit page (UI aligned with EMG Hardware Impact Tracker)
from __future__ import annotations

import html
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard.eeg_emg_dataset_quality import list_npz_files
from src.dashboard.eeg_emg_torch_hardware import (
    get_dashboard_system_context,
    list_saved_torch_hardware_reports,
    load_torch_hardware_payload,
    run_torch_profile,
    save_torch_hardware_report,
)
from src.eeg_emg.dashboard_defaults import (
    EEG_EMG_DATA_DIR,
    EEG_EMG_DEFAULT_NPZ,
    EEG_EMG_DEFAULT_PTH,
    EEG_EMG_MODEL_DIR,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG–EMG Hardware Impact | Umbra",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (match EMG hardware tracker) ─────────────────────────────────────────
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


def _fmt_mb(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:,.1f} MB"


def _fmt_params(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


def _report_as_dict(rep: object) -> dict:
    if isinstance(rep, dict):
        return rep
    if is_dataclass(rep):
        return asdict(rep)
    return {}


def _merged_system(rep: dict) -> dict:
    """Overlay snapshot from report onto live host stats (old reports may lack RAM keys)."""
    live = get_dashboard_system_context()
    snap = rep.get("system")
    if isinstance(snap, dict) and snap:
        return {**live, **snap}
    return live


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

st.sidebar.header("Checkpoint")
pth_files: list[str] = []
if not os.path.isdir(EEG_EMG_MODEL_DIR):
    st.sidebar.error("Model directory missing.")
else:
    pth_files = sorted(
        f for f in os.listdir(EEG_EMG_MODEL_DIR) if f.endswith(".pth") and not f.startswith(".")
    )

default_pth_name = Path(EEG_EMG_DEFAULT_PTH).name
pth_index = pth_files.index(default_pth_name) if default_pth_name in pth_files else 0
model_file: str | None = None
if pth_files:
    model_file = st.sidebar.selectbox("Model", pth_files, index=pth_index)

st.sidebar.header("Profiling")
bs_str = st.sidebar.text_input("Batch sizes (comma-separated)", value="1, 4, 8, 16")
n_runs = st.sidebar.number_input("Batches per size", 1, 100, 5)
eeg_fs = st.sidebar.number_input(
    "EEG fs (Hz)",
    1.0,
    10000.0,
    250.0,
    help="Used to compute real-time factor (window duration = window_size / fs).",
)
step = st.sidebar.number_input("Window step", 1, 2048, 128)
val_ratio = st.sidebar.slider("Validation fraction", 0.05, 0.5, 0.2, 0.05)
no_cuda = st.sidebar.checkbox("Force CPU", value=False)
run_label = st.sidebar.text_input("Report label", value="profile")

_ctx = get_dashboard_system_context()
st.sidebar.caption(
    f"Torch {_ctx.get('torch_version', '?')} · "
    f"{'CUDA ' + str(_ctx.get('cuda_version')) if _ctx.get('cuda_available') else 'CPU only'}"
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
    st.session_state["eeg_emg_hw_fs"] = float(eeg_fs)
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

# ── Page header ───────────────────────────────────────────────────────────────
st.title("EEG–EMG Hardware Impact Tracker")
st.caption(
    "PyTorch checkpoint profiling: load cost, model footprint, inference latency, "
    "throughput, and real-time factor (RTF)."
)

rep_raw = st.session_state.get("eeg_emg_hw")
if rep_raw is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">&nbsp;</div>
            <div class="empty-title">No profile run yet</div>
            <div class="empty-body">
                Select a dataset and checkpoint in the sidebar, set batch sizes, then click
                <strong>Run profile</strong>, or load a saved report.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

rep = _report_as_dict(rep_raw)
if not isinstance(rep, dict):
    st.error("Invalid report in session.")
    st.stop()

fs = float(st.session_state.get("eeg_emg_hw_fs", eeg_fs))
loading = rep.get("loading") or {}
inference = rep.get("inference") or []
cfg = rep.get("config") or {}

if loading.get("load_error"):
    st.error(loading["load_error"])
    st.stop()

_sys = _merged_system(rep)
_has_gpu = bool(_sys.get("cuda_available"))
gpu_label = html.escape(str(_sys.get("device_name", "CPU")))
gpu_badge = (
    '<span class="badge badge-gpu">CUDA</span>'
    if _has_gpu
    else '<span class="badge badge-cpu">CPU</span>'
)

# ── 1 · System information ───────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">System information</p>', unsafe_allow_html=True)

si1, si2, si3, si4, si5 = st.columns(5)
si1.metric("Platform", str(_sys.get("platform_str", "—"))[:30])
si2.metric(
    "CPU cores",
    f"{_sys.get('cpu_count_physical', '?')} phys / {_sys.get('cpu_count_logical', '?')} logical",
)
si3.metric("RAM (total)", _fmt_mb(_sys.get("ram_total_mb")))
si4.metric("RAM (free at snapshot)", _fmt_mb(_sys.get("ram_available_mb")))
si5.metric("PyTorch", str(_sys.get("torch_version", "—")))

st.markdown(
    f"**Device:** {gpu_label} &nbsp; {gpu_badge} &nbsp;&nbsp; "
    f"**Python:** {html.escape(str(_sys.get('python_version', '—')))} &nbsp;&nbsp; "
    f"**CUDA:** {html.escape(str(_sys.get('cuda_version') or '—'))}",
    unsafe_allow_html=True,
)

# ── 2 · Model loading profile ────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Model loading profile</p>', unsafe_allow_html=True)

ram_before = float(loading.get("ram_before_mb") or 0)
ram_after = float(loading.get("ram_after_mb") or 0)
ram_delta = ram_after - ram_before
n_params = loading.get("n_params_total")

l1, l2, l3, l4, l5 = st.columns(5)
l1.metric("Load time", f"{loading.get('load_time_s', 0):.3f} s")
l2.metric("File size", _fmt_mb(loading.get("file_size_mb")))
l3.metric("Parameters", _fmt_params(n_params))
l4.metric(
    "RAM delta (load)",
    _fmt_mb(ram_delta),
    help="Process RSS after load minus before load (includes Python / PyTorch overhead).",
)
cm = loading.get("cuda_mem_after_mb")
l5.metric(
    "CUDA alloc (after load)",
    _fmt_mb(cm) if cm is not None else ("N/A" if not _has_gpu else "—"),
)

with st.expander("Full loading detail", expanded=False):
    rows = [
        ("Model file", loading.get("model_file", "—")),
        ("Load time (s)", f"{loading.get('load_time_s', 0):.4f}"),
        ("File size (MB)", f"{loading.get('file_size_mb', 0):.3f}"),
        ("Parameters", _fmt_params(n_params)),
        ("RAM before (MB)", f"{ram_before:.1f}"),
        ("RAM after (MB)", f"{ram_after:.1f}"),
        ("RAM delta (MB)", f"{ram_delta:+.1f}"),
        (
            "CUDA after (MB)",
            f"{cm:.1f}" if cm is not None else "—",
        ),
        ("EEG channels", str(loading.get("n_eeg_channels", "—"))),
        ("EMG channels", str(loading.get("n_emg_channels", "—"))),
        ("Window size", str(loading.get("window_size", "—"))),
        ("Profile device", str(cfg.get("device", "—"))),
        ("Generated at", str(rep.get("generated_at", "—"))),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Metric", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

# ── 3 · Model architecture ──────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Model architecture</p>', unsafe_allow_html=True)

n_total = loading.get("n_params_total")
n_train = loading.get("n_params_trainable")
ws = loading.get("window_size") or cfg.get("window_size")

ca, cb, cc, cd = st.columns(4)
if n_total is not None:
    ca.metric("Total params", f"{n_total:,}")
if n_train is not None:
    cb.metric("Trainable params", f"{n_train:,}")
if n_total is not None and n_train is not None:
    cc.metric("Non-trainable", f"{n_total - n_train:,}")
if ws:
    cd.metric("Window size (samples)", ws)

c_eeg, c_emg, c_pmb, c_fs_col = st.columns(4)
c_eeg.metric("EEG channels", loading.get("n_eeg_channels", "—"))
c_emg.metric("EMG channels", loading.get("n_emg_channels", "—"))
if n_total:
    approx_mb = n_total * 4 / (1024 * 1024)
    c_pmb.metric("Params (float32 MB)", f"{approx_mb:.2f}")
c_fs_col.metric("EEG fs for RTF (Hz)", fs)

# ── 4 · Batch inference profile ─────────────────────────────────────────────
window_duration_s = (ws / fs) if ws and fs > 0 else None

if inference:
    st.divider()
    st.markdown('<p class="section-title">Batch inference profile</p>', unsafe_allow_html=True)
    ds_name = Path(str(rep.get("npz_path", ""))).name or "—"
    st.caption(
        f"Dataset: **{ds_name}** · **{cfg.get('n_batch_runs', n_runs)}** timed batches per size · "
        f"validation fraction **{cfg.get('val_ratio', val_ratio)}** · "
        "CUDA runs call `torch.cuda.synchronize()` before stopping the timer."
    )

    rows = []
    for row in inference:
        mean_ms = float(row["mean_ms"])
        bs = int(row["batch_size"])
        mean_s = mean_ms / 1000.0
        throughput = bs / mean_s if mean_s > 0 else None
        rtf = (mean_s / window_duration_s) if window_duration_s and window_duration_s > 0 else None
        per_sample_ms = mean_ms / bs if bs > 0 else None
        rows.append(
            {
                "Batch size": bs,
                "Mean (ms)": round(mean_ms, 2),
                "Std (ms)": round(row["std_ms"], 2),
                "p50 (ms)": round(row["p50_ms"], 2),
                "p95 (ms)": round(row["p95_ms"], 2),
                "Per window (ms)": round(per_sample_ms, 3) if per_sample_ms is not None else None,
                "Throughput (win/s)": round(throughput, 1) if throughput else None,
                "RTF": round(rtf, 4) if rtf else None,
            }
        )
    df = pd.DataFrame(rows)

    def _rtf_color(val: float | None) -> str:
        if val is None:
            return ""
        if val < 1.0:
            return "background-color: rgba(0,180,80,0.2)"
        return "background-color: rgba(220,50,50,0.2)"

    _bar_props = {"cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4}

    tab_table, tab_latency, tab_throughput = st.tabs(["Table", "Latency vs batch", "Throughput"])

    with tab_table:
        st.dataframe(
            df.style.applymap(_rtf_color, subset=["RTF"]).format(
                {
                    "RTF": lambda v: f"{v:.4f}" if v is not None else "—",
                    "Throughput (win/s)": lambda v: f"{v:.1f}" if v is not None else "—",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Batch size": st.column_config.NumberColumn("Batch size", format="%d"),
                "Mean (ms)": st.column_config.NumberColumn("Mean (ms)", format="%.2f"),
                "Std (ms)": st.column_config.NumberColumn("Std (ms)", format="%.2f"),
                "p50 (ms)": st.column_config.NumberColumn("p50 (ms)", format="%.2f"),
                "p95 (ms)": st.column_config.NumberColumn("p95 (ms)", format="%.2f"),
                "Per window (ms)": st.column_config.NumberColumn("Per window (ms)", format="%.3f"),
                "Throughput (win/s)": st.column_config.ProgressColumn(
                    "Throughput (win/s)",
                    min_value=0,
                    max_value=float(df["Throughput (win/s)"].max() or 1),
                    format="%.1f",
                ),
            },
        )
        if window_duration_s:
            st.caption(
                f"RTF = latency / window duration ({window_duration_s * 1000:.1f} ms). "
                "**RTF < 1.0** (green) = faster than real-time at the EEG fs you set."
            )

    with tab_latency:
        lat_long = []
        for _, b in df.iterrows():
            bs = str(int(b["Batch size"]))
            for metric, val in [
                ("Mean", b["Mean (ms)"]),
                ("p50", b["p50 (ms)"]),
                ("p95", b["p95 (ms)"]),
            ]:
                lat_long.append({"Batch size": bs, "Metric": metric, "ms": float(val)})
        df_lat = pd.DataFrame(lat_long)
        chart_lat = (
            alt.Chart(df_lat)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "Batch size:O",
                    sort=[str(int(x)) for x in df["Batch size"].tolist()],
                    title="Batch size",
                ),
                y=alt.Y("ms:Q", title="Latency (ms)"),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(scheme="tableau10"),
                    sort=["Mean", "p50", "p95"],
                ),
                tooltip=["Batch size", "Metric", alt.Tooltip("ms:Q", format=".2f")],
            )
            .properties(height=360, title="Latency vs batch size (lower is better)")
        )
        err_rows = [
            {
                "Batch size": str(int(r["Batch size"])),
                "lo": max(0.0, r["Mean (ms)"] - r["Std (ms)"]),
                "hi": r["Mean (ms)"] + r["Std (ms)"],
            }
            for _, r in df.iterrows()
        ]
        df_err = pd.DataFrame(err_rows)
        order_bs = [str(int(x)) for x in df["Batch size"].tolist()]
        err_band = (
            alt.Chart(df_err)
            .mark_area(opacity=0.15, color="#60a5fa")
            .encode(
                x=alt.X("Batch size:O", sort=order_bs),
                y=alt.Y("lo:Q"),
                y2=alt.Y2("hi:Q"),
            )
        )
        st.altair_chart(err_band + chart_lat, use_container_width=True)
        st.caption("Shaded band = mean ± 1σ across timed batches.")

        st.markdown("**Per-window latency** (batch mean ÷ batch size):")
        df_pw = df[["Batch size", "Per window (ms)"]].copy()
        best_idx = df["Per window (ms)"].idxmin()
        best_bs = int(df.loc[best_idx, "Batch size"])
        st.dataframe(df_pw, use_container_width=True, hide_index=True)
        st.caption(
            f"Most efficient batch size by mean time per window: **{best_bs}** "
            f"({df.loc[df['Batch size'] == best_bs, 'Per window (ms)'].values[0]:.3f} ms/window)."
        )

    with tab_throughput:
        df_tp = df[["Batch size", "Throughput (win/s)"]].copy()
        df_tp["Batch size"] = df_tp["Batch size"].astype(str)
        chart_tp = (
            alt.Chart(df_tp)
            .mark_bar(**_bar_props)
            .encode(
                x=alt.X(
                    "Batch size:O",
                    sort=[str(int(x)) for x in df["Batch size"].tolist()],
                    title="Batch size",
                ),
                y=alt.Y("Throughput (win/s):Q", title="Windows / second"),
                color=alt.Color(
                    "Throughput (win/s):Q",
                    scale=alt.Scale(scheme="greens"),
                    legend=None,
                ),
                tooltip=[
                    "Batch size",
                    alt.Tooltip("Throughput (win/s):Q", format=".1f"),
                ],
            )
            .properties(height=360, title="Throughput vs batch size (higher is better)")
        )
        label_layer = chart_tp + chart_tp.mark_text(
            align="center", dy=-10, fontSize=11, fontWeight="bold", color="#e2e8f0"
        ).encode(text=alt.Text("Throughput (win/s):Q", format=".0f"))
        st.altair_chart(label_layer, use_container_width=True)
        st.caption("Throughput = windows processed per second (batch_size / mean_batch_latency).")
else:
    st.divider()
    st.warning("No inference timings (empty validation set or profiling error).")
    df = pd.DataFrame()

# ── 5 · Summary ───────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Summary</p>', unsafe_allow_html=True)

summary_rows: list[tuple[str, str]] = [
    ("Model", str(rep.get("model_file", "—"))),
    ("Dataset", Path(str(rep.get("npz_path", ""))).name or "—"),
    ("Generated at", str(rep.get("generated_at", "—"))),
    ("PyTorch", str(_sys.get("torch_version", "—"))),
    ("Device", str(_sys.get("device_name", "—"))),
    ("Load time", f"{loading.get('load_time_s', 0):.3f} s"),
    ("File size", _fmt_mb(loading.get("file_size_mb"))),
    ("Parameters", _fmt_params(n_params)),
    ("RAM delta (load)", _fmt_mb(ram_delta)),
]

if inference and not df.empty:
    fastest = df.loc[df["Per window (ms)"].idxmin()]
    fastest_tput = df.loc[df["Throughput (win/s)"].idxmax()]
    bs1 = df[df["Batch size"] == 1]
    summary_rows += [
        (
            "Latency batch size 1 (mean ms)",
            f"{bs1['Mean (ms)'].values[0]:.2f}" if len(bs1) else "—",
        ),
        (
            "Best mean ms/window",
            f"bs={int(fastest['Batch size'])} ({fastest['Per window (ms)']:.3f} ms)",
        ),
        (
            "Peak throughput",
            f"{fastest_tput['Throughput (win/s)']:.1f} win/s (bs={int(fastest_tput['Batch size'])})",
        ),
    ]

st.dataframe(
    pd.DataFrame(summary_rows, columns=["Metric", "Value"]),
    use_container_width=True,
    hide_index=True,
)
