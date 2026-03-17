#!/usr/bin/env python3
"""Headless model benchmark runner for CI.

Evaluates all (or specified) Keras models, saves a full JSON report to
data/comparison_reports/, appends a row per model to a persistent CSV history
file, and writes a Markdown summary to $GITHUB_STEP_SUMMARY when running in
GitHub Actions.

The JSON report is saved with the same format as the dashboard's "Save
comparison" feature, so it can be loaded directly from the Model Comparator
page without any conversion.

Usage:
    python scripts/run_comparison.py [options]

Examples:
    python scripts/run_comparison.py --dataset-id 1
    python scripts/run_comparison.py --dataset-id 1 --n-windows 300 --n-timing 20
    python scripts/run_comparison.py --models cnn_lstm_emg_v3.keras --dataset-id 1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make src.* importable when script is run from any working directory.
_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo))

from src.dashboard.model_comparator import (  # noqa: E402
    ModelComparisonResult,
    get_available_models,
    run_model_comparison,
    save_comparison,
)

# ── CSV schema ────────────────────────────────────────────────────────────────

_HISTORY_FILE = "ci_history.csv"
_HISTORY_FIELDS = [
    "timestamp",
    "git_sha",
    "git_ref",
    "dataset_id",
    "n_windows",
    "model_name",
    "accuracy",
    "top3_accuracy",
    "macro_f1",
    "mean_inference_ms",
    "std_inference_ms",
    "file_size_mb",
    "param_count",
    "is_broken",
    "broken_reasons",
]


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CI model benchmark — compare all models and persist results."
    )
    p.add_argument(
        "--dataset-id",
        default=None,
        metavar="ID",
        help="Preprocessed dataset ID (e.g. 1). Omit for metadata-only mode.",
    )
    p.add_argument(
        "--n-windows",
        type=int,
        default=500,
        metavar="N",
        help="Dataset windows to evaluate per model (default: 500).",
    )
    p.add_argument(
        "--n-timing",
        type=int,
        default=30,
        metavar="N",
        help="Single-window predict() calls per model for latency (default: 30).",
    )
    p.add_argument(
        "--models",
        nargs="*",
        metavar="MODEL",
        default=None,
        help="Explicit .keras filenames. Defaults to all models in src/models/.",
    )
    p.add_argument(
        "--output-dir",
        default="data/comparison_reports",
        metavar="DIR",
        help="Directory for JSON report and history CSV (default: data/comparison_reports).",
    )
    return p.parse_args()


# ── Formatting helpers ────────────────────────────────────────────────────────


def _pct(v: float | None) -> str:
    return f"{v * 100:.2f}%" if v is not None else "—"


def _ms(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "—"


def _params(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ── GitHub Step Summary ───────────────────────────────────────────────────────


def _write_github_summary(
    results: list[ModelComparisonResult],
    dataset_id: str | None,
    n_windows: int,
    git_sha: str,
    report_path: Path,
) -> None:
    """Append a Markdown benchmark table to $GITHUB_STEP_SUMMARY."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    lines = [
        "## ⚡ Model Benchmark Report\n",
        f"**Commit:** `{git_sha[:8]}`  \n"
        f"**Dataset:** `{dataset_id or 'metadata only'}`  \n"
        f"**Windows evaluated:** {n_windows}  \n"
        f"**Report file:** `{report_path.name}`\n",
        "| Model | Status | Accuracy | Top-3 | Macro F1 | Latency (ms ± σ) | Params | Size (MB) |",
        "|:------|:------:|--------:|------:|--------:|----------------:|------:|--------:|",
    ]

    for r in results:
        if not r.loaded:
            status = "❌ failed"
        elif r.is_broken:
            status = "⚠️ broken"
        elif r.accuracy is None:
            status = "ℹ️ meta"
        else:
            status = "✅ ok"

        lat = (
            f"{r.mean_inference_ms:.1f} ± {r.std_inference_ms:.1f}"
            if r.mean_inference_ms is not None
            else "—"
        )
        lines.append(
            f"| `{r.name.replace('.keras', '')}` "
            f"| {status} "
            f"| {_pct(r.accuracy)} "
            f"| {_pct(r.top3_accuracy)} "
            f"| {_pct(r.macro_f1)} "
            f"| {lat} "
            f"| {_params(r.param_count)} "
            f"| {r.file_size_mb:.2f} |"
        )

    with open(summary_file, "a") as f:
        f.write("\n".join(lines) + "\n\n")


# ── History CSV ───────────────────────────────────────────────────────────────


def _append_history(
    results: list[ModelComparisonResult],
    output_dir: Path,
    dataset_id: str | None,
    n_windows: int,
    git_sha: str,
    git_ref: str,
) -> Path:
    """Append one row per model to the persistent CI history CSV."""
    history_path = output_dir / _HISTORY_FILE
    new_file = not history_path.exists()

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017

    with open(history_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_HISTORY_FIELDS)
        if new_file:
            writer.writeheader()

        for r in results:
            writer.writerow(
                {
                    "timestamp": ts,
                    "git_sha": git_sha,
                    "git_ref": git_ref,
                    "dataset_id": dataset_id or "",
                    "n_windows": n_windows,
                    "model_name": r.name,
                    "accuracy": round(r.accuracy, 6) if r.accuracy is not None else "",
                    "top3_accuracy": round(r.top3_accuracy, 6)
                    if r.top3_accuracy is not None
                    else "",
                    "macro_f1": round(r.macro_f1, 6) if r.macro_f1 is not None else "",
                    "mean_inference_ms": round(r.mean_inference_ms, 3)
                    if r.mean_inference_ms is not None
                    else "",
                    "std_inference_ms": round(r.std_inference_ms, 3)
                    if r.std_inference_ms is not None
                    else "",
                    "file_size_mb": round(r.file_size_mb, 4),
                    "param_count": r.param_count or "",
                    "is_broken": r.is_broken,
                    "broken_reasons": "; ".join(r.broken_reasons),
                }
            )

    return history_path


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()

    git_sha = os.environ.get("GITHUB_SHA", "local")
    git_ref = os.environ.get("GITHUB_REF_NAME", "local")

    models = args.models or get_available_models()
    if not models:
        print("[SKIP] No .keras models found in src/models/", file=sys.stderr)
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Models    : {', '.join(models)}")
    print(f"Dataset   : {args.dataset_id or 'none (metadata only)'}")
    print(f"Windows   : {args.n_windows}  |  Timing calls: {args.n_timing}")
    print(f"Git SHA   : {git_sha[:8]}  |  Ref: {git_ref}")
    print()

    results = run_model_comparison(
        model_names=models,
        dataset_id=args.dataset_id,
        n_windows=args.n_windows,
        n_timing_samples=args.n_timing,
    )

    # ── Print summary table ────────────────────────────────────────────────────
    col = "{:<28} {:<10} {:>10} {:>10} {:>14} {:>10}"
    header = col.format("Model", "Status", "Accuracy", "Macro F1", "Latency (ms)", "Params")
    print(header)
    print("─" * len(header))

    broken_count = 0
    for r in results:
        if not r.loaded:
            status = "LOAD-FAIL"
        elif r.is_broken:
            status = "BROKEN"
            broken_count += 1
        elif r.accuracy is None:
            status = "META-ONLY"
        else:
            status = "OK"

        print(
            col.format(
                r.name.replace(".keras", ""),
                status,
                _pct(r.accuracy),
                _pct(r.macro_f1),
                _ms(r.mean_inference_ms),
                _params(r.param_count),
            )
        )

    print()

    # ── Save full JSON report ──────────────────────────────────────────────────
    run_label = f"ci_{git_sha[:8]}"
    config = {
        "dataset_id": args.dataset_id,
        "n_windows": args.n_windows,
        "n_timing": args.n_timing,
        "use_dataset": args.dataset_id is not None,
        "git_sha": git_sha,
        "git_ref": git_ref,
    }
    report_path = save_comparison(results, config, name=run_label)
    print(f"Report    : {report_path}")

    # ── Append to history CSV ──────────────────────────────────────────────────
    history_path = _append_history(
        results, output_dir, args.dataset_id, args.n_windows, git_sha, git_ref
    )
    print(f"History   : {history_path}")

    # ── GitHub Step Summary ────────────────────────────────────────────────────
    _write_github_summary(results, args.dataset_id, args.n_windows, git_sha, report_path)

    if broken_count:
        print(f"\n⚠️  {broken_count} model(s) flagged as broken.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
