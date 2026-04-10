"""Plot strategy comparison from sliding-month Sharpe series (same setup as months_validate).

Generates PNG figures under the output directory:
  - sharpe_by_holdout_month.png   — lines vs calendar month index
  - sharpe_boxplot.png           — distribution per strategy
  - mean_sharpe_bar.png          — mean ± std across holdout months
  - sharpe_heatmap.png           — strategy × holdout month
  - mean_vs_std_scatter.png      — one point per strategy

Usage:
    python plot_strategy_comparison.py              # all strategies in STRATEGY_REGISTRY
    python plot_strategy_comparison.py -o figures/
    python plot_strategy_comparison.py baseline intraday_risk_parity   # subset only

If you only pass two names, every chart will show just those two (by design).

Note: Long-only diversified strategies often have **highly correlated** holdout-month
Sharpes (~0.95+) because each month the same market path drives most PnL; weights can
differ while Sharpe-through-time lines still look similar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_strategies_months import evaluate_strategies_sharpes
from months_validate import TRAIN_MONTHS_PER_BLOCK, sliding_month_groups_shape
from submission import N_ASSETS, STRATEGY_REGISTRY, load_meta, load_prices


def _pairwise_sharpe_correlations(sharpes: dict[str, np.ndarray]) -> np.ndarray:
    """Pearson correlation of holdout-month Sharpe vectors between each pair of strategies."""
    names = sorted(sharpes.keys())
    k = len(names)
    if k < 2:
        return np.array([])
    m = np.column_stack([sharpes[n] for n in names])
    return np.corrcoef(m.T)


def _holdout_month_indices(n_groups: int) -> np.ndarray:
    return np.arange(n_groups, dtype=int) + TRAIN_MONTHS_PER_BLOCK


def plot_all(
    sharpes: dict[str, np.ndarray],
    out_dir: Path,
    *,
    dpi: int = 150,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(sharpes.keys())
    if not names:
        raise ValueError("No strategy results to plot")

    n_groups = sharpes[names[0]].shape[0]
    for n in names[1:]:
        if sharpes[n].shape[0] != n_groups:
            raise ValueError("All strategies must have the same number of holdout months")

    x_month = _holdout_month_indices(n_groups)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(10, len(names))))[: len(names)]

    # 1) Lines: Sharpe vs holdout month
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        ax.plot(x_month, sharpes[name], label=name, color=cmap[i], linewidth=1.5, alpha=0.9)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Holdout month index (0-based timeline)")
    ax.set_ylabel("Annualized Sharpe (holdout month)")
    ax.set_title("Sharpe by holdout month - sliding 24m train / 1m OOS")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "sharpe_by_holdout_month.png", dpi=dpi)
    plt.close(fig)

    # 2) Box plot
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [sharpes[n] for n in names]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap[i])
        patch.set_alpha(0.55)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Annualized Sharpe")
    ax.set_title("Distribution of holdout-month Sharpes")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "sharpe_boxplot.png", dpi=dpi)
    plt.close(fig)

    # 3) Bar: mean ± std
    means = np.array([float(np.mean(sharpes[n])) for n in names])
    stds = (
        np.array([float(np.std(sharpes[n], ddof=1)) for n in names])
        if n_groups >= 2
        else np.zeros(len(names))
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds if n_groups >= 2 else None, capsize=4, color=cmap[: len(names)], alpha=0.75, ecolor="black")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Mean Sharpe (± std across months)")
    ax.set_title("Mean holdout Sharpe with dispersion")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_sharpe_bar.png", dpi=dpi)
    plt.close(fig)

    # 4) Heatmap: strategy × month
    mat = np.vstack([sharpes[n] for n in names])
    vmax = float(np.nanmax(np.abs(mat)))
    if not np.isfinite(vmax) or vmax < 1e-9:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(max(8, n_groups * 0.22), max(4, len(names) * 0.45)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    xt_idx = np.arange(0, n_groups, max(1, n_groups // 12))
    ax.set_xticks(xt_idx)
    ax.set_xticklabels([str(x_month[i]) for i in xt_idx])
    ax.set_xlabel("Holdout month index")
    ax.set_title("Sharpe heatmap (strategy × holdout month)")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(out_dir / "sharpe_heatmap.png", dpi=dpi)
    plt.close(fig)

    # 5) Scatter: mean vs std
    fig, ax = plt.subplots(figsize=(7, 6))
    if n_groups >= 2:
        for i, name in enumerate(names):
            ax.scatter(stds[i], means[i], s=80, color=cmap[i], edgecolors="black", linewidths=0.5, zorder=3)
            ax.annotate(name, (stds[i], means[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("Std (Sharpe across holdout months)")
        ax.set_ylabel("Mean Sharpe")
        ax.set_title("Mean vs dispersion of monthly Sharpes")
        ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
        ax.axvline(0.0, color="gray", linewidth=0.6, linestyle="--")
    else:
        ax.text(0.5, 0.5, "Need at least 2 holdout months", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_vs_std_scatter.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sliding-month strategy comparison.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for PNG files (default: figures/)",
    )
    parser.add_argument(
        "strategies",
        nargs="*",
        metavar="NAME",
        help="Strategy keys (default: all in registry)",
    )
    args = parser.parse_args()

    print("Loading data...")
    prices = load_prices()
    meta = load_meta()
    assert prices.shape[1] == N_ASSETS, f"Expected {N_ASSETS} assets, got {prices.shape[1]}"

    sliding_month_groups_shape(prices)

    all_registered = sorted(STRATEGY_REGISTRY.keys())
    names = list(args.strategies) if args.strategies else all_registered
    unknown = [n for n in names if n not in STRATEGY_REGISTRY]
    if unknown:
        print("Unknown strategy name(s):", ", ".join(unknown), file=sys.stderr)
        print("Available:", ", ".join(all_registered), file=sys.stderr)
        raise SystemExit(1)

    if args.strategies:
        print(
            f"Subset: evaluating {len(names)} of {len(all_registered)} strateg(ies): {', '.join(names)}"
        )
        print(
            f"  (Run with no strategy names to compare all {len(all_registered)}: "
            f"python plot_strategy_comparison.py -o figures)"
        )
    else:
        print(f"Full comparison: all {len(names)} registered strateg(ies).")

    print("Evaluating strategies (this may take a while)...")
    sharpes, errors, n_groups = evaluate_strategies_sharpes(prices, meta, names)
    for name, msg in errors.items():
        if name not in sharpes:
            print(f"  skip {name!r}: {msg}")

    if not sharpes:
        raise SystemExit("No successful strategy evaluations; nothing to plot.")

    corr = _pairwise_sharpe_correlations(sharpes)
    if corr.size > 0 and len(sharpes) >= 2:
        off_diag = corr[np.triu_indices_from(corr, k=1)]
        print(
            f"Holdout-month Sharpe correlation across strategies: "
            f"min={float(np.min(off_diag)):.3f}, max={float(np.max(off_diag)):.3f}, "
            f"mean={float(np.mean(off_diag)):.3f} (often high for long-only portfolios on the same months)"
        )

    print(f"Plotting {len(sharpes)} strateg(ies) -> {args.output_dir.resolve()}")
    plot_all(sharpes, args.output_dir)
    print("Done:")
    for fn in sorted(args.output_dir.glob("*.png")):
        print(f"  {fn}")


if __name__ == "__main__":
    main()
