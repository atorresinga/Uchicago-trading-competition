"""Visualize grid_search_intraday_results.csv.

Produces (under -o, default figures/):
  grid_search_bars.png       — every run as a horizontal bar, sorted by mean Sharpe
  grid_search_heatmaps.png  — 2x2 facets: cov_lookback x blend_rate for each (vol_lookback, vol_target)
  grid_search_scatter.png   — mean_sharpe vs std_sharpe, points labeled by key params

Usage:
    python plot_grid_search_results.py
    python plot_grid_search_results.py --csv grid_search_intraday_results.csv -o figures/
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_CHOICES = ("mean_sharpe", "score", "std_sharpe")


def _varying_param_columns(df: pd.DataFrame) -> list[str]:
    skip = {"mean_sharpe", "std_sharpe", "score", "n_blowups", "n_holdout_months"}
    out = []
    for c in df.columns:
        if c in skip:
            continue
        if df[c].nunique(dropna=False) > 1:
            out.append(c)
    return out


def _short_row_label(row: pd.Series, cols: list[str], max_len: int = 55) -> str:
    parts = [f"{c[:5]}={row[c]}" for c in cols]
    s = " | ".join(parts)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def vary_cols(df: pd.DataFrame) -> list[str]:
    return _varying_param_columns(df)


def plot_sorted_bars(df: pd.DataFrame, out_path: Path, metric: str, dpi: int) -> None:
    vc = vary_cols(df)
    df = df.copy()
    df["_label"] = df.apply(lambda r: _short_row_label(r, vc), axis=1)
    df = df.sort_values(metric, ascending=True)
    n = len(df)
    fig_h = max(6.0, 0.22 * n + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(n)
    colors = plt.cm.RdYlGn(0.15 + 0.7 * (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min() + 1e-12))
    ax.barh(y, df[metric], color=colors, height=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["_label"], fontsize=7)
    ax.set_xlabel(metric)
    ax.set_title(f"Grid search: all {n} runs (sorted by {metric})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_faceted_heatmaps(df: pd.DataFrame, out_path: Path, metric: str, dpi: int) -> None:
    """2x2 panels: vol_lookback x vol_target facets; each panel cov x blend heatmap."""
    need = {"cov_lookback_days", "blend_rate", "vol_lookback_days", "vol_target"}
    if not need.issubset(df.columns):
        return
    vls = sorted(df["vol_lookback_days"].unique())
    vts = sorted(df["vol_target"].unique())
    pairs = list(itertools.product(vls, vts))
    n = len(pairs)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)
    vmin = float(df[metric].min())
    vmax = float(df[metric].max())
    im = None
    for idx, (vl, vt) in enumerate(pairs):
        ax = axes.flat[idx]
        sub = df[(df["vol_lookback_days"] == vl) & (df["vol_target"] == vt)]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot_table(
            index="cov_lookback_days",
            columns="blend_rate",
            values=metric,
            aggfunc="mean",
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(x) for x in pivot.columns], rotation=15)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(x) for x in pivot.index])
        ax.set_xlabel("blend_rate")
        ax.set_ylabel("cov_lookback_days")
        ax.set_title(f"vol_lb={vl}, vol_tgt={vt}")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="black")
    for idx in range(len(pairs), len(axes.flat)):
        axes.flat[idx].set_visible(False)
    if im is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=metric)
    fig.suptitle(f"{metric}: cov × blend (faceted by vol_lookback & vol_target)", y=1.02)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_mean_vs_std(df: pd.DataFrame, out_path: Path, metric: str, dpi: int) -> None:
    if "std_sharpe" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    x = df["std_sharpe"].values
    y = df[metric].values
    sc = ax.scatter(x, y, c=df[metric], cmap="RdYlGn", s=120, edgecolors="black", linewidths=0.4)
    vc = vary_cols(df)
    for pos, (_, row) in enumerate(df.iterrows()):
        lab = _short_row_label(row, vc, max_len=40)
        ax.annotate(lab, (x[pos], y[pos]), textcoords="offset points", xytext=(4, 3), fontsize=5, alpha=0.85)
    ax.set_xlabel("std_sharpe (across holdout months)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs dispersion (each point is one grid row)")
    fig.colorbar(sc, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot grid search CSV results.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("grid_search_intraday_results.csv"),
        help="Input CSV from grid_search_intraday.py",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for PNG files",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default="mean_sharpe",
        help="Color / bar length metric (default: mean_sharpe)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"File not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        raise SystemExit(f"Column {args.metric!r} missing in CSV")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Varying params: {vary_cols(df)}")

    plot_sorted_bars(df, args.output_dir / "grid_search_bars.png", args.metric, args.dpi)
    print(f"  wrote {args.output_dir / 'grid_search_bars.png'}")

    if {"cov_lookback_days", "blend_rate", "vol_lookback_days", "vol_target"}.issubset(df.columns):
        plot_faceted_heatmaps(df, args.output_dir / "grid_search_heatmaps.png", args.metric, args.dpi)
        print(f"  wrote {args.output_dir / 'grid_search_heatmaps.png'}")

    if "std_sharpe" in df.columns:
        plot_mean_vs_std(df, args.output_dir / "grid_search_scatter.png", args.metric, args.dpi)
        print(f"  wrote {args.output_dir / 'grid_search_scatter.png'}")

    print("Done.")


if __name__ == "__main__":
    main()
