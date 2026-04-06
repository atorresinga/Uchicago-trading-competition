from __future__ import annotations

from typing import Callable

"""Participant submission scaffold for the portfolio optimization case.

Implement your strategy by modifying the MyStrategy class below.
The default is equal-weight (1/N) across all 25 assets.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


N_ASSETS = 25
TICKS_PER_DAY = 30
ASSET_COLUMNS = tuple(f"A{i:02d}" for i in range(N_ASSETS))


@dataclass(frozen=True)
class PublicMeta:
    """Per-asset metadata visible to participants."""

    sector_id: np.ndarray
    spread_bps: np.ndarray
    borrow_bps_annual: np.ndarray


def load_prices(path: str = "prices.csv") -> np.ndarray:
    """Load the price matrix from CSV. Returns shape (n_ticks, 25)."""
    df = pd.read_csv(path, index_col="tick")
    return df[list(ASSET_COLUMNS)].to_numpy(dtype=float)


def load_meta(path: str = "meta.csv") -> PublicMeta:
    """Load asset metadata from CSV."""
    df = pd.read_csv(path)
    return PublicMeta(
        sector_id=df["sector_id"].to_numpy(dtype=int),
        spread_bps=df["spread_bps"].to_numpy(dtype=float),
        borrow_bps_annual=df["borrow_bps_annual"].to_numpy(dtype=float),
    )


class StrategyBase:
    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        pass

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        raise NotImplementedError


class Baseline(StrategyBase):
    """Equal-weight baseline."""

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        return np.ones(N_ASSETS) / N_ASSETS


# ---------------------------------------------------------------------------
# Risk Parity Strategy
# ---------------------------------------------------------------------------

def _daily_returns(prices: np.ndarray) -> np.ndarray:
    """Tick prices -> daily close-to-close simple returns. Shape (n_days-1, 25)."""
    n_days = prices.shape[0] // TICKS_PER_DAY
    closes = prices[np.arange(1, n_days + 1) * TICKS_PER_DAY - 1]
    # Use log returns then convert back — avoids divide-by-zero on near-zero prices
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rets = np.log(closes[1:] / closes[:-1])
    # Replace any inf/nan with 0 (asset didn't move or had bad data)
    log_rets = np.nan_to_num(log_rets, nan=0.0, posinf=0.0, neginf=0.0)
    return np.expm1(log_rets)  # convert back to simple returns


def _intraday_tick_returns(prices: np.ndarray) -> np.ndarray:
    """Tick prices -> tick-to-tick log returns. Shape (n_ticks-1, 25)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rets = np.log(prices[1:] / prices[:-1])
    log_rets = np.nan_to_num(log_rets, nan=0.0, posinf=0.0, neginf=0.0)
    return log_rets


def _realized_vol_intraday(prices: np.ndarray, lookback_days: int) -> np.ndarray:
    """Compute annualized realized volatility per asset using intraday tick returns.

    Uses the last `lookback_days` days of tick data. Much more precise than
    daily close-to-close vol because we have 30 observations per day.

    Returns shape (25,) — annualized vol per asset.
    """
    n_ticks = prices.shape[0]
    lookback_ticks = lookback_days * TICKS_PER_DAY
    start = max(0, n_ticks - lookback_ticks)
    recent_prices = prices[start:]

    tick_rets = _intraday_tick_returns(recent_prices)

    # Realized variance = sum of squared returns per day, then average across days
    # This is the standard realized variance estimator
    n_rets = tick_rets.shape[0]
    n_days_actual = n_rets // TICKS_PER_DAY
    if n_days_actual < 1:
        return np.ones(prices.shape[1]) * 0.3  # fallback

    # Reshape into (n_days, ticks_per_day, n_assets), compute daily RV
    usable = n_days_actual * TICKS_PER_DAY
    reshaped = tick_rets[-usable:].reshape(n_days_actual, TICKS_PER_DAY, -1)
    daily_rv = np.sum(reshaped ** 2, axis=1)  # (n_days, 25)

    # Average daily RV, then annualize
    avg_daily_rv = np.mean(daily_rv, axis=0)
    ann_vol = np.sqrt(avg_daily_rv * 252)

    ann_vol = np.maximum(ann_vol, 1e-12)
    return ann_vol


def _realized_covariance_intraday(prices: np.ndarray, lookback_days: int) -> np.ndarray:
    """Compute realized covariance matrix using intraday tick returns.

    Much more statistically efficient than daily covariance — 30x more observations.
    Uses Ledoit-Wolf shrinkage on tick returns for numerical stability.

    Returns shape (25, 25) covariance matrix (annualized).
    """
    n_ticks = prices.shape[0]
    lookback_ticks = lookback_days * TICKS_PER_DAY
    start = max(0, n_ticks - lookback_ticks)
    recent_prices = prices[start:]

    tick_rets = _intraday_tick_returns(recent_prices)

    if tick_rets.shape[0] < 60:
        return np.eye(prices.shape[1]) * 0.01

    # Standardize before LedoitWolf
    stds = np.std(tick_rets, axis=0)
    valid = stds > 1e-12

    if np.sum(valid) < 2:
        return np.eye(prices.shape[1]) * 0.01

    standardized = tick_rets[:, valid] / stds[valid][np.newaxis, :]

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = LedoitWolf().fit(standardized).covariance_
        if not np.all(np.isfinite(corr)):
            return np.eye(prices.shape[1]) * 0.01
    except Exception:
        return np.eye(prices.shape[1]) * 0.01

    # Rescale to covariance and annualize
    # Tick returns have variance ~ daily_var / 30, so multiply by 30 * 252
    cov = corr * np.outer(stds[valid], stds[valid]) * TICKS_PER_DAY * 252

    # Map back to full matrix
    full_cov = np.eye(prices.shape[1]) * 0.01
    idx = np.where(valid)[0]
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            full_cov[ii, jj] = cov[i, j]

    return full_cov


def _risk_parity_weights(cov: np.ndarray, max_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
    """Iterative fixed-point solver for risk parity weights.

    Finds w such that each asset's risk contribution w_i * (Sigma w)_i is equal.
    Update rule: w_i <- 1 / (Sigma w)_i, then normalize to sum to 1.
    """
    n = cov.shape[0]

    # Regularize: add small ridge to diagonal to ensure positive-definiteness
    cov = cov.copy()
    cov += np.eye(n) * (1e-8 * np.trace(cov) / n)

    w = np.ones(n) / n

    for _ in range(max_iter):
        sigma_w = cov @ w
        # Clamp marginal risk contributions away from zero
        sigma_w = np.clip(sigma_w, 1e-10, None)
        w_new = 1.0 / sigma_w
        w_sum = np.sum(w_new)
        if w_sum < 1e-12:
            return np.ones(n) / n  # fallback to equal weight
        w_new /= w_sum

        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new

    return w


class RiskParity(StrategyBase):
    """Risk Parity with Ledoit-Wolf shrunk covariance."""

    def __init__(self, lookback: int = 120, rebalance_freq: int = 10, blend_rate: float = 1.0):
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.blend_rate = blend_rate
        self.current_weights = np.ones(N_ASSETS) / N_ASSETS
        self.last_rebalance_day = -999

    def _compute_target(self, daily_rets: np.ndarray) -> np.ndarray:
        """Estimate covariance -> solve risk parity -> return long-only weights."""
        recent = daily_rets[-self.lookback:] if daily_rets.shape[0] > self.lookback else daily_rets

        # Guard against degenerate data (constant columns, etc.)
        stds = np.std(recent, axis=0)
        valid = stds > 1e-12

        if np.sum(valid) < 2:
            return np.ones(N_ASSETS) / N_ASSETS

        recent_valid = recent[:, valid]
        valid_stds = stds[valid]

        # Standardize returns before LedoitWolf to prevent overflow in scipy matmul
        # Correlation matrix is numerically much more stable than raw covariance
        standardized = recent_valid / valid_stds[np.newaxis, :]

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr = LedoitWolf().fit(standardized).covariance_
            # If scipy overflowed, the result will contain inf/nan
            if not np.all(np.isfinite(corr)):
                return np.ones(N_ASSETS) / N_ASSETS
        except Exception:
            return np.ones(N_ASSETS) / N_ASSETS

        # Rescale correlation back to covariance: Cov = diag(std) @ Corr @ diag(std)
        cov_full = corr * np.outer(valid_stds, valid_stds)

        w_valid = _risk_parity_weights(cov_full)

        # Map back to full 25-asset weight vector
        w = np.zeros(N_ASSETS)
        w[valid] = w_valid

        w = np.maximum(w, 0.0)  # long-only
        w_sum = np.sum(w)
        if w_sum > 1e-12:
            w /= w_sum
        else:
            w = np.ones(N_ASSETS) / N_ASSETS
        return w

    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        """Pre-compute initial weights from training data."""
        daily_rets = _daily_returns(train_prices)
        self.current_weights = self._compute_target(daily_rets)

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        """Return weights. Only recompute every rebalance_freq days."""
        # Skip rebalance if not time yet
        if day > 0 and (day - self.last_rebalance_day) < self.rebalance_freq:
            return self.current_weights

        daily_rets = _daily_returns(price_history)
        if daily_rets.shape[0] < 30:
            return self.current_weights

        target = self._compute_target(daily_rets)

        # Blend: 1.0 = full jump, <1.0 = partial move toward target
        self.current_weights = self.blend_rate * target + (1.0 - self.blend_rate) * self.current_weights

        if np.sum(np.abs(self.current_weights)) > 1.0:
            self.current_weights /= np.sum(np.abs(self.current_weights))

        self.last_rebalance_day = day
        return self.current_weights


# ---------------------------------------------------------------------------
# Volatility-Targeted Risk Parity Strategy
# ---------------------------------------------------------------------------

class VolTargetRiskParity(StrategyBase):
    """Risk Parity base allocation with volatility targeting overlay.

    Computes risk parity weights, then scales total portfolio exposure
    inversely to recent realized volatility. When vol is high, reduce
    exposure (sum of |w| < 1, effectively holding cash). When vol is
    low, use full exposure.

    Hyperparameters:
        lookback:       days for covariance estimation (risk parity)
        rebalance_freq: days between weight recomputation
        blend_rate:     partial adjustment toward new target weights
        vol_lookback:   days for realized vol calculation
        vol_target:     target annualized portfolio vol (scaling anchor)
        vol_cap:        max scaling factor (1.0 = never lever up)
        vol_floor:      min scaling factor (prevents going to zero exposure)
    """

    def __init__(
        self,
        lookback: int = 20,
        rebalance_freq: int = 20,
        blend_rate: float = 1.0,
        vol_lookback: int = 20,
        vol_target: float = 0.15,
        vol_cap: float = 1.0,
        vol_floor: float = 0.2,
    ):
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.blend_rate = blend_rate
        self.vol_lookback = vol_lookback
        self.vol_target = vol_target
        self.vol_cap = vol_cap
        self.vol_floor = vol_floor

        self.current_weights = np.ones(N_ASSETS) / N_ASSETS
        self.base_weights = np.ones(N_ASSETS) / N_ASSETS  # unscaled risk parity weights
        self.last_rebalance_day = -999

    def _compute_base_weights(self, daily_rets: np.ndarray) -> np.ndarray:
        """Risk parity weights (unscaled, sum to 1)."""
        recent = daily_rets[-self.lookback:] if daily_rets.shape[0] > self.lookback else daily_rets

        stds = np.std(recent, axis=0)
        valid = stds > 1e-12

        if np.sum(valid) < 2:
            return np.ones(N_ASSETS) / N_ASSETS

        recent_valid = recent[:, valid]
        valid_stds = stds[valid]

        standardized = recent_valid / valid_stds[np.newaxis, :]

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr = LedoitWolf().fit(standardized).covariance_
            if not np.all(np.isfinite(corr)):
                return np.ones(N_ASSETS) / N_ASSETS
        except Exception:
            return np.ones(N_ASSETS) / N_ASSETS

        cov_full = corr * np.outer(valid_stds, valid_stds)
        w_valid = _risk_parity_weights(cov_full)

        w = np.zeros(N_ASSETS)
        w[valid] = w_valid
        w = np.maximum(w, 0.0)
        w_sum = np.sum(w)
        if w_sum > 1e-12:
            w /= w_sum
        else:
            w = np.ones(N_ASSETS) / N_ASSETS
        return w

    def _compute_vol_scalar(self, daily_rets: np.ndarray) -> float:
        """Compute scaling factor based on recent realized portfolio vol.

        scalar = vol_target / realized_vol, clamped to [vol_floor, vol_cap].
        """
        if daily_rets.shape[0] < self.vol_lookback:
            return 1.0

        # Portfolio returns using current base weights
        recent_rets = daily_rets[-self.vol_lookback:]
        port_rets = recent_rets @ self.base_weights
        realized_vol = np.std(port_rets) * np.sqrt(252)

        if realized_vol < 1e-12:
            return self.vol_cap

        scalar = self.vol_target / realized_vol
        return np.clip(scalar, self.vol_floor, self.vol_cap)

    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        daily_rets = _daily_returns(train_prices)
        self.base_weights = self._compute_base_weights(daily_rets)
        scalar = self._compute_vol_scalar(daily_rets)
        self.current_weights = self.base_weights * scalar

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        daily_rets = _daily_returns(price_history)
        if daily_rets.shape[0] < 30:
            return self.current_weights

        # Recompute base weights on rebalance days
        if day == 0 or (day - self.last_rebalance_day) >= self.rebalance_freq:
            target_base = self._compute_base_weights(daily_rets)
            self.base_weights = (self.blend_rate * target_base
                                 + (1.0 - self.blend_rate) * self.base_weights)
            self.last_rebalance_day = day

        # Always update vol scalar (cheap, no turnover cost if base weights don't change)
        scalar = self._compute_vol_scalar(daily_rets)
        self.current_weights = self.base_weights * scalar

        # Enforce gross exposure <= 1
        gross = np.sum(np.abs(self.current_weights))
        if gross > 1.0:
            self.current_weights /= gross

        return self.current_weights


# ---------------------------------------------------------------------------
# Cost-Aware Static Tilt Strategy
# ---------------------------------------------------------------------------

class CostAwareTilt(StrategyBase):
    """Static allocation that tilts toward high-Sharpe, low-cost assets.

    Computes historical Sharpe for each asset over training data, then
    weights proportional to a score that rewards high Sharpe and penalizes
    high spread and borrow costs. Allocation is set once in fit() and
    never changes — zero turnover, zero transaction costs after entry.

    Hyperparameters:
        sharpe_weight:   how much to reward historical Sharpe (exponent)
        spread_penalty:  how much to penalize high spread assets
        borrow_penalty:  how much to penalize high borrow cost assets
        min_weight:      minimum weight per asset (0 = can fully exclude)
        zero_negative:   if True, zero out assets with negative historical Sharpe
    """

    def __init__(
        self,
        sharpe_weight: float = 1.0,
        spread_penalty: float = 1.0,
        borrow_penalty: float = 0.0,
        min_weight: float = 0.0,
        zero_negative: bool = True,
    ):
        self.sharpe_weight = sharpe_weight
        self.spread_penalty = spread_penalty
        self.borrow_penalty = borrow_penalty
        self.min_weight = min_weight
        self.zero_negative = zero_negative
        self.weights = np.ones(N_ASSETS) / N_ASSETS

    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        daily_rets = _daily_returns(train_prices)

        # Per-asset annualized Sharpe
        mu = np.mean(daily_rets, axis=0)
        sigma = np.std(daily_rets, axis=0)
        sigma = np.maximum(sigma, 1e-12)
        asset_sharpes = (mu / sigma) * np.sqrt(252)

        # Normalize spread and borrow to [0, 1] range for comparable penalties
        spread_norm = meta.spread_bps / np.max(meta.spread_bps)
        borrow_norm = meta.borrow_bps_annual / np.max(meta.borrow_bps_annual)

        # Score = Sharpe^weight - penalties
        scores = np.copy(asset_sharpes)

        # Optionally zero out negative Sharpe assets
        if self.zero_negative:
            scores = np.maximum(scores, 0.0)

        # Apply Sharpe exponent (higher weight = more concentrated in winners)
        scores = np.sign(scores) * np.abs(scores) ** self.sharpe_weight

        # Subtract cost penalties
        scores -= self.spread_penalty * spread_norm
        scores -= self.borrow_penalty * borrow_norm

        # Floor at min_weight equivalent score
        scores = np.maximum(scores, self.min_weight)

        # Normalize to sum to 1
        total = np.sum(scores)
        if total > 1e-12:
            self.weights = scores / total
        else:
            self.weights = np.ones(N_ASSETS) / N_ASSETS

        # Enforce minimum weight
        if self.min_weight > 0:
            self.weights = np.maximum(self.weights, self.min_weight)
            self.weights /= np.sum(self.weights)

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        return self.weights


# ---------------------------------------------------------------------------
# Rolling Momentum Tilt Strategy
# ---------------------------------------------------------------------------

class MomentumTilt(StrategyBase):
    """Adaptive momentum-based allocation with cost awareness.

    Instead of a static tilt from training history, computes rolling
    momentum over recent data and rebalances periodically. Adapts to
    regime changes because the signal updates as new data arrives.

    Weights are proportional to a score combining:
      - Rolling momentum (recent cumulative return over lookback window)
      - Cost penalties (spread, borrow)

    Hyperparameters:
        momentum_lookback:  days of recent returns for momentum signal
        rebalance_freq:     days between rebalances
        spread_penalty:     penalty for high-spread assets
        borrow_penalty:     penalty for high-borrow assets
        zero_negative:      if True, zero out assets with negative momentum
        blend_rate:         partial adjustment toward new target (1.0 = full)
        equal_weight_blend: fraction of equal weight to mix in (0 = pure momentum)
    """

    def __init__(
        self,
        momentum_lookback: int = 60,
        rebalance_freq: int = 20,
        spread_penalty: float = 1.0,
        borrow_penalty: float = 2.0,
        zero_negative: bool = True,
        blend_rate: float = 1.0,
        equal_weight_blend: float = 0.0,
    ):
        self.momentum_lookback = momentum_lookback
        self.rebalance_freq = rebalance_freq
        self.spread_penalty = spread_penalty
        self.borrow_penalty = borrow_penalty
        self.zero_negative = zero_negative
        self.blend_rate = blend_rate
        self.equal_weight_blend = equal_weight_blend

        self.current_weights = np.ones(N_ASSETS) / N_ASSETS
        self.last_rebalance_day = -999
        self.spread_norm = None
        self.borrow_norm = None

    def _compute_target(self, daily_rets: np.ndarray) -> np.ndarray:
        """Compute target weights from rolling momentum + cost penalties."""
        # Rolling momentum: cumulative return over lookback window
        if daily_rets.shape[0] >= self.momentum_lookback:
            recent = daily_rets[-self.momentum_lookback:]
        else:
            recent = daily_rets

        # Cumulative return over the window (product of 1+r)
        cum_rets = np.prod(1.0 + recent, axis=0) - 1.0

        # Momentum score: use cumulative return
        scores = cum_rets.copy()

        # Zero out negative momentum assets
        if self.zero_negative:
            scores = np.maximum(scores, 0.0)

        # Subtract cost penalties
        if self.spread_norm is not None:
            scores -= self.spread_penalty * self.spread_norm
        if self.borrow_norm is not None:
            scores -= self.borrow_penalty * self.borrow_norm

        # Floor at zero
        scores = np.maximum(scores, 0.0)

        # Normalize
        total = np.sum(scores)
        if total > 1e-12:
            w = scores / total
        else:
            w = np.ones(N_ASSETS) / N_ASSETS

        # Blend with equal weight for diversification
        if self.equal_weight_blend > 0:
            ew = np.ones(N_ASSETS) / N_ASSETS
            w = (1.0 - self.equal_weight_blend) * w + self.equal_weight_blend * ew

        return w

    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        # Pre-compute normalized costs (fixed throughout)
        self.spread_norm = meta.spread_bps / np.max(meta.spread_bps)
        self.borrow_norm = meta.borrow_bps_annual / np.max(meta.borrow_bps_annual)

        # Compute initial weights from training data
        daily_rets = _daily_returns(train_prices)
        self.current_weights = self._compute_target(daily_rets)

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        # Skip if not rebalance day
        if day > 0 and (day - self.last_rebalance_day) < self.rebalance_freq:
            return self.current_weights

        daily_rets = _daily_returns(price_history)
        if daily_rets.shape[0] < 20:
            return self.current_weights

        target = self._compute_target(daily_rets)

        # Blend toward target
        self.current_weights = (self.blend_rate * target
                                + (1.0 - self.blend_rate) * self.current_weights)

        # Normalize
        gross = np.sum(np.abs(self.current_weights))
        if gross > 1.0:
            self.current_weights /= gross

        self.last_rebalance_day = day
        return self.current_weights


# ---------------------------------------------------------------------------
# Intraday Risk Parity Strategy
# ---------------------------------------------------------------------------

class IntradayRiskParity(StrategyBase):
    """Risk Parity using intraday tick data for covariance and vol estimation.

    Key improvements over basic RiskParity:
    - Uses tick-level realized variance (30x more data points per day)
    - More responsive vol targeting from intraday realized vol
    - Optional cost-aware tilt on top of risk parity base
    - Vol targeting scales total exposure inversely to recent vol

    Hyperparameters:
        cov_lookback_days:  days of tick data for covariance estimation
        vol_lookback_days:  days of tick data for vol targeting scalar
        rebalance_freq:     days between base weight recomputation
        blend_rate:         partial adjustment toward new base weights
        vol_target:         target annualized portfolio vol
        vol_cap:            max exposure scaling (1.0 = no leverage)
        vol_floor:          min exposure scaling
        use_intraday_cov:   if True, use tick-level cov; if False, daily cov
        spread_penalty:     penalize high-spread assets in weight calc
        borrow_penalty:     penalize high-borrow assets in weight calc
    """

    def __init__(
        self,
        cov_lookback_days: int = 60,
        vol_lookback_days: int = 20,
        rebalance_freq: int = 20,
        blend_rate: float = 0.5,
        vol_target: float = 0.15,
        vol_cap: float = 1.0,
        vol_floor: float = 0.2,
        use_intraday_cov: bool = True,
        spread_penalty: float = 0.5,
        borrow_penalty: float = 1.0,
    ):
        self.cov_lookback_days = cov_lookback_days
        self.vol_lookback_days = vol_lookback_days
        self.rebalance_freq = rebalance_freq
        self.blend_rate = blend_rate
        self.vol_target = vol_target
        self.vol_cap = vol_cap
        self.vol_floor = vol_floor
        self.use_intraday_cov = use_intraday_cov
        self.spread_penalty = spread_penalty
        self.borrow_penalty = borrow_penalty

        self.current_weights = np.ones(N_ASSETS) / N_ASSETS
        self.base_weights = np.ones(N_ASSETS) / N_ASSETS
        self.last_rebalance_day = -999
        self.spread_norm = None
        self.borrow_norm = None

    def _compute_base_weights(self, prices: np.ndarray) -> np.ndarray:
        """Compute risk parity weights using intraday covariance."""
        if self.use_intraday_cov:
            cov = _realized_covariance_intraday(prices, self.cov_lookback_days)
        else:
            daily_rets = _daily_returns(prices)
            recent = daily_rets[-self.cov_lookback_days:]
            if recent.shape[0] < 20:
                return np.ones(N_ASSETS) / N_ASSETS
            stds = np.std(recent, axis=0)
            valid = stds > 1e-12
            if np.sum(valid) < 2:
                return np.ones(N_ASSETS) / N_ASSETS
            standardized = recent[:, valid] / stds[valid][np.newaxis, :]
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr = LedoitWolf().fit(standardized).covariance_
                if not np.all(np.isfinite(corr)):
                    return np.ones(N_ASSETS) / N_ASSETS
            except Exception:
                return np.ones(N_ASSETS) / N_ASSETS
            cov = np.eye(N_ASSETS) * 0.01
            idx = np.where(valid)[0]
            for i, ii in enumerate(idx):
                for j, jj in enumerate(idx):
                    cov[ii, jj] = corr[i, j] * stds[ii] * stds[jj]

        # Risk parity on the covariance
        w = _risk_parity_weights(cov)
        w = np.maximum(w, 0.0)

        # Apply cost penalties to shift weight toward cheaper assets
        if self.spread_norm is not None and self.spread_penalty > 0:
            cost_adj = 1.0 - self.spread_penalty * self.spread_norm
            cost_adj = np.maximum(cost_adj, 0.1)
            w *= cost_adj

        if self.borrow_norm is not None and self.borrow_penalty > 0:
            borrow_adj = 1.0 - self.borrow_penalty * self.borrow_norm
            borrow_adj = np.maximum(borrow_adj, 0.1)
            w *= borrow_adj

        # Normalize
        w_sum = np.sum(w)
        if w_sum > 1e-12:
            w /= w_sum
        else:
            w = np.ones(N_ASSETS) / N_ASSETS
        return w

    def _compute_vol_scalar(self, prices: np.ndarray) -> float:
        """Scale exposure based on intraday realized vol."""
        ann_vol = _realized_vol_intraday(prices, self.vol_lookback_days)

        # Portfolio vol using current base weights
        port_vol = np.sqrt(np.sum((self.base_weights * ann_vol) ** 2))

        if port_vol < 1e-12:
            return self.vol_cap

        scalar = self.vol_target / port_vol
        return float(np.clip(scalar, self.vol_floor, self.vol_cap))

    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        self.spread_norm = meta.spread_bps / np.max(meta.spread_bps)
        self.borrow_norm = meta.borrow_bps_annual / np.max(meta.borrow_bps_annual)

        self.base_weights = self._compute_base_weights(train_prices)
        scalar = self._compute_vol_scalar(train_prices)
        self.current_weights = self.base_weights * scalar

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        n_ticks = price_history.shape[0]
        if n_ticks < TICKS_PER_DAY * 30:
            return self.current_weights

        # Recompute base weights on rebalance days
        if day == 0 or (day - self.last_rebalance_day) >= self.rebalance_freq:
            target_base = self._compute_base_weights(price_history)
            self.base_weights = (self.blend_rate * target_base
                                 + (1.0 - self.blend_rate) * self.base_weights)
            self.last_rebalance_day = day

        # Always update vol scalar (uses recent ticks, very responsive)
        scalar = self._compute_vol_scalar(price_history)
        self.current_weights = self.base_weights * scalar

        # Enforce gross exposure <= 1
        gross = np.sum(np.abs(self.current_weights))
        if gross > 1.0:
            self.current_weights /= gross

        return self.current_weights


def create_default_intraday_risk_parity() -> IntradayRiskParity:
    """Default tuned IntradayRiskParity (same hyperparameters as competition submission)."""
    return IntradayRiskParity(
        cov_lookback_days=60,
        vol_lookback_days=20,
        rebalance_freq=20,
        blend_rate=0.5,
        vol_target=0.15,
        vol_cap=1.0,
        vol_floor=0.2,
        use_intraday_cov=True,
        spread_penalty=0.5,
        borrow_penalty=1.0,
    )


# Factories for batch comparison (months_validate / compare_strategies_months).
STRATEGY_REGISTRY: dict[str, Callable[[], StrategyBase]] = {
    "baseline": Baseline,
    "risk_parity": RiskParity,
    "vol_target_risk_parity": VolTargetRiskParity,
    "cost_aware_tilt": CostAwareTilt,
    "momentum_tilt": MomentumTilt,
    "intraday_risk_parity": create_default_intraday_risk_parity,
}


def create_strategy() -> StrategyBase:
    """Entry point called by validate.py and months_validate.py."""
    return create_default_intraday_risk_parity()
