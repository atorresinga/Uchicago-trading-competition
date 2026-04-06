from __future__ import annotations

from typing import Callable

"""Local evaluator: sliding 24+1 months with ``validate.py`` scoring.

**Scoring:** Same as ``validate.py`` — ``run_backtest`` and helpers are unchanged.

**Splits:** Trading months are ``252/12`` days × ``TICKS_PER_DAY`` ticks. For
``j = 0, 1, …`` we train on months ``j .. j+23`` (24 months; training **may
overlap** between consecutive ``j``) and hold out **only** month ``j+24``. Holdout
months are disjoint (24, 25, …, up to the last full month in the data). Example
with 60 months: last group is train ``35..58``, hold month ``59``.

Each group uses a fresh strategy. We report every holdout Sharpe, then mean and
variance of those Sharpes.

Usage:
    python months_validate.py
"""

import math

import numpy as np

from submission import PublicMeta, StrategyBase, create_strategy, load_meta, load_prices

# ---------------------------------------------------------------------------
# Constants (must match the competition runtime)
# ---------------------------------------------------------------------------

N_ASSETS = 25
TICKS_PER_DAY = 30
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = TRADING_DAYS_PER_YEAR // 12  # ~21; equal-length "trading months"
TICKS_PER_MONTH = TRADING_DAYS_PER_MONTH * TICKS_PER_DAY
TRAIN_MONTHS_PER_BLOCK = 24
IMPACT_MULT = 2.5
DT_YEAR = 1.0 / (TRADING_DAYS_PER_YEAR * TICKS_PER_DAY)

# ---------------------------------------------------------------------------
# Evaluation helpers (copied from the competition runtime)
# ---------------------------------------------------------------------------


def project_to_gross_limit(w: np.ndarray) -> np.ndarray:
    """Project weights back onto the L1 gross-exposure constraint (<=1)."""
    w = np.asarray(w, dtype=float).copy()
    gross = float(np.sum(np.abs(w)))
    if not np.isfinite(gross):
        return w
    if gross > 1.0:
        w /= gross
    return w


def _transaction_cost(
    spread: np.ndarray, delta_weights: np.ndarray, impact_mult: float
) -> tuple[float, float]:
    """Linear and quadratic trading costs for a change in portfolio weights."""
    linear = float(np.sum((spread / 2.0) * np.abs(delta_weights)))
    quadratic = float(np.sum((impact_mult * spread) * (delta_weights**2)))
    return linear, quadratic


def _hold_fixed_weights_one_day(
    wealth: float,
    weights: np.ndarray,
    logret: np.ndarray,
    borrow: np.ndarray,
    *,
    day: int,
) -> float:
    """Advance wealth over one trading day while holding fixed portfolio weights."""
    t0 = day * TICKS_PER_DAY
    t_begin = t0 + 1 if day == 0 else t0
    for t in range(t_begin, t0 + TICKS_PER_DAY):
        pnl = float(np.sum(weights * (np.exp(logret[t]) - 1.0)))
        borrow_cost = float(np.sum(np.maximum(-weights, 0.0) * borrow) * DT_YEAR)
        wealth *= 1.0 + pnl - borrow_cost
    return wealth


def _history_through_day(
    train_prices: np.ndarray, hold_prices: np.ndarray, day: int
) -> np.ndarray:
    """History visible to the strategy after observing `day` holdout days."""
    cutoff = (day + 1) * TICKS_PER_DAY
    return np.vstack([train_prices, hold_prices[:cutoff]])


def annualized_sharpe(daily_returns: np.ndarray) -> float:
    """Annualized Sharpe ratio with zero risk-free rate."""
    x = np.asarray(daily_returns, dtype=float)
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd < 1e-12:
        return -np.inf if mu <= 0 else np.inf
    return math.sqrt(TRADING_DAYS_PER_YEAR) * mu / sd


def run_backtest(
    train_prices: np.ndarray,
    hold_prices: np.ndarray,
    strategy,
    meta: PublicMeta,
) -> dict:
    """Run the tick-level wealth process on the pseudo-holdout period."""
    spread = np.asarray(meta.spread_bps, dtype=float) / 1e4
    borrow = np.asarray(meta.borrow_bps_annual, dtype=float) / 1e4

    strategy.fit(train_prices, meta, ticks_per_day=TICKS_PER_DAY)
    weights = project_to_gross_limit(strategy.get_weights(train_prices, meta, day=0))
    assert np.all(np.isfinite(weights)), "Non-finite weights at initialization"

    wealth = 1.0
    entry_linear, entry_quadratic = _transaction_cost(spread, weights, IMPACT_MULT)
    wealth *= 1.0 - (entry_linear + entry_quadratic)

    logret = np.zeros_like(hold_prices)
    logret[1:] = np.log(hold_prices[1:] / hold_prices[:-1])

    n_days = hold_prices.shape[0] // TICKS_PER_DAY
    daily_returns = np.zeros(n_days)
    daily_costs = np.zeros(n_days + 1)
    daily_costs[0] = entry_linear + entry_quadratic

    for day in range(n_days):
        wealth_start = wealth
        try:
            wealth = _hold_fixed_weights_one_day(wealth, weights, logret, borrow, day=day)
        except FloatingPointError:
            wealth = float("nan")
        if wealth <= 0 or not np.isfinite(wealth):
            daily_returns[day:] = -1.0
            return {
                "daily_returns": daily_returns,
                "daily_costs": daily_costs[: day + 1],
                "blown_up": True,
            }

        history = _history_through_day(train_prices, hold_prices, day)
        target = project_to_gross_limit(strategy.get_weights(history, meta, day=day + 1))
        assert np.all(np.isfinite(target)), f"Non-finite weights on day {day}"

        delta = target - weights
        linear, quadratic = _transaction_cost(spread, delta, IMPACT_MULT)
        trade_cost = linear + quadratic
        wealth *= 1.0 - trade_cost
        daily_costs[day + 1] = trade_cost
        daily_returns[day] = wealth / wealth_start - 1.0
        weights = target

    return {
        "daily_returns": daily_returns,
        "daily_costs": daily_costs,
        "blown_up": False,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def sliding_month_groups_shape(prices: np.ndarray) -> tuple[int, int, int]:
    """Return (total_months, n_groups, tpm). Raises SystemExit if not enough data."""
    tpm = TICKS_PER_MONTH
    train_m = TRAIN_MONTHS_PER_BLOCK
    total_ticks = prices.shape[0]
    total_months = total_ticks // tpm
    j_max = total_months - (train_m + 1)
    n_groups = j_max + 1 if j_max >= 0 else 0
    if n_groups < 1:
        raise SystemExit(
            f"Need at least {train_m + 1} full trading months "
            f"({(train_m + 1) * tpm:,} ticks), have {total_months} months."
        )
    return total_months, n_groups, tpm


def evaluate_sliding_month_sharpes(
    prices: np.ndarray,
    meta: PublicMeta,
    strategy_factory: Callable[[], StrategyBase],
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Run sliding 24m train / 1m holdout; return annualized Sharpe per holdout month."""
    train_m = TRAIN_MONTHS_PER_BLOCK
    total_months, n_groups, tpm = sliding_month_groups_shape(prices)

    sharpes: list[float] = []
    for j in range(n_groups):
        train_start = j * tpm
        train_end = (j + train_m) * tpm
        hold_end = (j + train_m + 1) * tpm
        train_prices = prices[train_start:train_end]
        hold_prices = prices[train_end:hold_end]
        strat = strategy_factory()
        result = run_backtest(train_prices, hold_prices, strat, meta)
        sr = annualized_sharpe(result["daily_returns"])
        sharpes.append(sr)
        if verbose:
            hold_m = j + train_m
            blown = " (BLEW UP)" if result["blown_up"] else ""
            print(
                f"  Group {j + 1}/{n_groups}: train months {j}-{j + train_m - 1}, "
                f"hold month {hold_m}, Sharpe = {sr:+.4f}{blown}"
            )

    return np.asarray(sharpes, dtype=float)


def main() -> None:
    print("Loading data...")
    prices = load_prices()
    meta = load_meta()

    assert prices.shape[1] == N_ASSETS, f"Expected {N_ASSETS} assets, got {prices.shape[1]}"
    total_ticks = prices.shape[0]
    total_months, n_groups, _ = sliding_month_groups_shape(prices)

    print(f"  {total_ticks:,} ticks, {total_ticks // TICKS_PER_DAY} days, {N_ASSETS} assets")
    print(
        f"  ~{total_months} full trading months; {n_groups} group(s) "
        f"(sliding 24m train, disjoint 1m holdout)"
    )

    print("\nRunning backtests (train months j..j+23, hold month j+24)...")
    arr = evaluate_sliding_month_sharpes(prices, meta, create_strategy, verbose=True)

    print("\n" + "=" * 50)
    print("SHARPE ACROSS HOLDOUT MONTHS")
    print("=" * 50)
    print(f"  Mean Sharpe:     {float(np.mean(arr)):+.4f}")
    if n_groups >= 2:
        var = float(np.var(arr, ddof=1))
        print(f"  Var(Sharpe):     {var:.6f}")
        print(f"  Std(Sharpe):     {float(np.std(arr, ddof=1)):.4f}")
    else:
        print("  Var(Sharpe):     n/a (need at least 2 groups)")
    print("=" * 50)


if __name__ == "__main__":
    main()
