"""
case1_bot.py — UChicago xchange Case 1 trading bot (Python 3.12).

Strategies:
  1. Fundamental market-making on A (FV = EPS * 10), inventory-skewed.
  2. ETF/NAV arbitrage via swap_order (toETF / fromETF), 5-cash fee aware.
  3. Put-call parity arbitrage on B options at K in {950, 1000, 1050}.

Run:
    pip install git+https://github.com/UChicagoFM/utcxchangelib.git
    python case1_bot.py
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from utcxchangelib import XChangeClient, Side

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / f"round_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

def log_event(kind: str, **fields):
    """Append one JSON line. Cheap, fire-and-forget."""
    try:
        rec = {"ts": time.time(), "kind": kind, **fields}
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass  # never let logging break the bot

# ---------------------------------------------------------------------------
# Configuration — tune on competition day after limits are released.
# ---------------------------------------------------------------------------

SERVER   = "practice.uchicago.exchange:3333"
USERNAME = "mit_chicago"
PASSWORD = "xerus-nexus-lemon"

# ---- Strategy toggles (flip to False to isolate individual strategies) ----
ENABLE_STRAT_A       = False   # Fundamental market-making on A
ENABLE_STRAT_ETF     = True  # ETF / NAV arbitrage
ENABLE_STRAT_PARITY  = False  # Put-call parity on B options

# ---- Official risk limits (updated from competition announcement) ----------
MAX_POS: dict[str, int] = {
    "A": 200, "B": 200, "C": 200, "ETF": 200,
    "B_C_950": 200, "B_P_950": 200,
    "B_C_1000": 200, "B_P_1000": 200,
    "B_C_1050": 200, "B_P_1050": 200,
    "R_CUT": 200, "R_HOLD": 200, "R_HIKE": 200,
}
MAX_OPT_POS             = 200   # fallback for any unlisted symbol
MAX_ORDER_SIZE          = 40
MAX_OPEN_ORDERS         = 50
MAX_OUTSTANDING_VOLUME  = 120   # per-symbol limit; tracked to avoid silent rejects

# ---- C fair-value model parameters (disclosed by organizers) ---------------
# FV_C = C_BASE - lambda_ * D * B0_N * delta_y
#   where delta_y = rate_expectation_bps / 10_000
C_Y0      = 0.045   # base yield
C_PE0     = 14.0    # base P/E
C_EPS0    = 2.00    # base EPS
C_D       = 7.5     # bond-book duration
C_B0_N    = 40.0    # bond book per share at par
C_BASE    = 55.0    # base fair value at y0
C_LAMBDA  = 0.65    # rate-sensitivity weight

# ---- Strategy knobs --------------------------------------------------------
A_BASE_SPREAD  = 2       # ticks around fair value
A_QUOTE_SIZE   = 5
A_INV_SKEW     = 0.05    # price skew per unit of inventory
ETF_SWAP_FEE   = 5
ETF_ARB_BUFFER = 2       # extra cushion above the 5-cash fee
PARITY_BUFFER  = 2       # min edge over parity to act
LOOP_INTERVAL  = 0.5     # seconds between trade() iterations

OPTION_STRIKES = [950, 1000, 1050]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("case1")


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------

@dataclass
class FairValues:
    A: Optional[float] = None
    C: Optional[float] = None
    eps_A: Optional[float] = None
    cpi_actual: Optional[float] = None
    cpi_forecast: Optional[float] = None
    rate_expectation_bps: float = 0.0

    def recompute_C(self) -> None:
        """Update C fair value from current rate expectation."""
        delta_y = self.rate_expectation_bps / 10_000.0
        self.C = C_BASE - C_LAMBDA * C_D * C_B0_N * delta_y


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class Case1Bot(XChangeClient):
    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.fv = FairValues()
        self._reject_count = 0
        self._running = True

    # ---------- utility helpers ----------

    def _best_bid(self, sym: str) -> Optional[int]:
        book = self.order_books.get(sym)
        if not book or not book.bids:
            return None
        live = [p for p, q in book.bids.items() if q > 0]
        return max(live) if live else None

    def _best_ask(self, sym: str) -> Optional[int]:
        book = self.order_books.get(sym)
        if not book or not book.asks:
            return None
        live = [p for p, q in book.asks.items() if q > 0]
        return min(live) if live else None

    def _mid(self, sym: str) -> Optional[float]:
        b, a = self._best_bid(sym), self._best_ask(sym)
        if b is None or a is None:
            return None
        return (b + a) / 2.0

    def _pos(self, sym: str) -> int:
        return int(self.positions.get(sym, 0))

    def _room_to_trade(self, sym: str, qty: int, side: Side) -> int:
        """Return max tradeable qty without breaching position or order-size limits."""
        cap = MAX_POS.get(sym, MAX_OPT_POS)
        current = self._pos(sym)
        signed_room = cap - current if side == Side.BUY else cap + current
        return max(0, min(qty, signed_room, MAX_ORDER_SIZE))

    async def _safe_place(
        self, sym: str, qty: int, side: Side, price: Optional[int] = None
    ) -> Optional[str]:
        """Place an order with size clamp, position guard, and error handling."""
        try:
            clamped = self._room_to_trade(sym, qty, side)
            if clamped <= 0:
                return None
            if len(self.open_orders) >= MAX_OPEN_ORDERS:
                return None
            return await self.place_order(sym, clamped, side, price)
        except Exception as e:
            log.error("place_order failed %s %s %s @ %s: %s", sym, qty, side, price, e)
            return None

    async def _cancel_orders_for(self, sym: Optional[str] = None) -> None:
        """Cancel all open orders, optionally filtered to a single symbol."""
        for oid, order_info in list(self.open_orders.items()):
            if sym is None or (order_info and order_info[0].symbol == sym):
                try:
                    await self.cancel_order(oid)
                except Exception as e:
                    log.error("cancel %s failed: %s", oid, e)

    # ---------- event handlers ----------

    async def bot_handle_cancel_response(self, order_id, success, error=None):
        if not success:
            log.warning("Cancel failed %s: %s", order_id, error)

    async def bot_handle_order_fill(self, order_id, qty, price):
        log.info("FILL %s qty=%s px=%s pos=%s", order_id, qty, price,
                 dict(self.positions))
        log_event("fill",
                  order_id=order_id, qty=qty, price=price,
                  positions=dict(self.positions))

    async def bot_handle_order_rejected(self, order_id, reason):
        self._reject_count += 1
        log.warning("REJECT %s: %s (total=%d)", order_id, reason,
                    self._reject_count)
        log_event("reject", order_id=order_id, reason=reason)

    async def bot_handle_trade_msg(self, symbol, price, qty):
        pass

    async def bot_handle_book_update(self, symbol):
        pass

    async def bot_handle_swap_response(self, swap, qty, success):
        log.info("SWAP %s qty=%s success=%s", swap, qty, success)
        log_event("swap", swap=swap, qty=qty, success=success)

    async def bot_handle_news(self, news_release: dict):
        """
        Parse structured news. A can receive multiple EPS releases per day —
        each one immediately updates FV_A. CPI prints re-price the rate complex.
        """
        try:
            kind = news_release.get("kind")
            data = news_release.get("new_data") or {}

            log_event("news", news_kind=kind, data=data)

            if kind != "structured":
                return

            sub = data.get("structured_subtype")

            if sub == "earnings" and data.get("asset") == "A":
                eps = float(data.get("value", 0.0))
                self.fv.eps_A = eps
                self.fv.A = eps * 10.0  # P/E = 10 fixed
                log.info("EARNINGS A: EPS=%.4f -> FV_A=%.2f", eps, self.fv.A)

            elif sub == "cpi_print":
                self.fv.cpi_forecast = float(data.get("forecast", 0.0))
                self.fv.cpi_actual   = float(data.get("actual", 0.0))
                log.info("CPI: forecast=%.3f actual=%.3f",
                         self.fv.cpi_forecast, self.fv.cpi_actual)
                # CPI surprise shifts rate expectations -> re-price C immediately
                self.fv.recompute_C()

        except Exception:
            log.error("news handler error:\n%s", traceback.format_exc())

    async def bot_handle_market_resolved(self, market_id, winning_symbol, tick):
        log.info("Prediction market %s resolved: %s at tick %s",
                 market_id, winning_symbol, tick)
        # Prediction market resolution can cause a discrete jump in C's bond leg
        self.fv.recompute_C()

    async def bot_handle_settlement_payout(self, user, market_id, amount, tick):
        log.info("Payout %s from market %s at tick %s", amount, market_id, tick)

    # ---------- rate signal from prediction market ----------

    def _update_rate_expectation(self) -> None:
        """E[dr] = 25*q_HIKE - 25*q_CUT (bps), from prediction-market mids."""
        mids: dict[str, float] = {}
        for sym in ("R_HIKE", "R_HOLD", "R_CUT"):
            m = self._mid(sym)
            if m is None:
                return
            mids[sym] = max(0.0, min(1.0, m / 100.0))

        total = sum(mids.values()) or 1.0
        q = {k: v / total for k, v in mids.items()}
        new_bps = 25.0 * q["R_HIKE"] - 25.0 * q["R_CUT"]

        if new_bps != self.fv.rate_expectation_bps:
            self.fv.rate_expectation_bps = new_bps
            self.fv.recompute_C()
            log.debug("Rate expectation updated: %.2f bps -> FV_C=%.2f",
                      new_bps, self.fv.C)

    # ---------- Strategy 1: market-make A around fundamental FV ----------

    async def strat_market_make_A(self) -> None:
        if self.fv.A is None:
            return

        fv   = self.fv.A
        inv  = self._pos("A")
        skew = A_INV_SKEW * inv  # long -> shift quotes down

        bid_px = int(round(fv - A_BASE_SPREAD - skew))
        ask_px = int(round(fv + A_BASE_SPREAD - skew))
        if ask_px <= bid_px:
            ask_px = bid_px + 1

        await self._cancel_orders_for("A")
        await self._safe_place("A", A_QUOTE_SIZE, Side.BUY,  bid_px)
        await self._safe_place("A", A_QUOTE_SIZE, Side.SELL, ask_px)

    # ---------- Strategy 2: ETF / NAV arbitrage ----------

    async def strat_etf_nav_arb(self) -> None:
        a_bid, a_ask = self._best_bid("A"), self._best_ask("A")
        b_bid, b_ask = self._best_bid("B"), self._best_ask("B")
        c_bid, c_ask = self._best_bid("C"), self._best_ask("C")
        etf_bid      = self._best_bid("ETF")
        etf_ask      = self._best_ask("ETF")

        if None in (a_bid, a_ask, b_bid, b_ask, c_bid, c_ask, etf_bid, etf_ask):
            return

        nav_buy  = a_ask + b_ask + c_ask   # cost to buy all three components
        nav_sell = a_bid + b_bid + c_bid   # proceeds from selling all three
        threshold = ETF_SWAP_FEE + ETF_ARB_BUFFER

        # ETF rich: sell ETF on book, create via toETF (components -> ETF)
        if etf_bid - nav_buy > threshold:
            await self._safe_place("ETF", 1, Side.SELL, etf_bid)
            try:
                await self.place_swap_order("toETF", 1)
            except Exception as e:
                log.error("toETF swap failed: %s", e)

        # ETF cheap: buy ETF on book, redeem via fromETF (ETF -> components)
        elif nav_sell - etf_ask > threshold:
            await self._safe_place("ETF", 1, Side.BUY, etf_ask)
            try:
                await self.place_swap_order("fromETF", 1)
            except Exception as e:
                log.error("fromETF swap failed: %s", e)

    # ---------- Strategy 3: put-call parity on B options ----------

    async def strat_parity_B(self) -> None:
        s = self._mid("B")
        if s is None:
            return

        for K in OPTION_STRIKES:
            call_sym = f"B_C_{K}"
            put_sym  = f"B_P_{K}"

            c_mid = self._mid(call_sym)
            p_mid = self._mid(put_sym)
            if c_mid is None or p_mid is None:
                continue

            # Parity: C - P = S - K  (r ~ 0)
            edge = (c_mid - p_mid) - (s - K)

            if edge > PARITY_BUFFER:
                # Call rich, put cheap: sell call, buy put, buy stock
                c_bid = self._best_bid(call_sym)
                p_ask = self._best_ask(put_sym)
                b_ask = self._best_ask("B")
                if None in (c_bid, p_ask, b_ask):
                    continue
                await self._safe_place(call_sym, 1, Side.SELL, c_bid)
                await self._safe_place(put_sym,  1, Side.BUY,  p_ask)
                await self._safe_place("B",      1, Side.BUY,  b_ask)

            elif -edge > PARITY_BUFFER:
                # Put rich, call cheap: buy call, sell put, sell stock
                c_ask = self._best_ask(call_sym)
                p_bid = self._best_bid(put_sym)
                b_bid = self._best_bid("B")
                if None in (c_ask, p_bid, b_bid):
                    continue
                await self._safe_place(call_sym, 1, Side.BUY,  c_ask)
                await self._safe_place(put_sym,  1, Side.SELL, p_bid)
                await self._safe_place("B",      1, Side.SELL, b_bid)

    # ---------- main loop ----------

    async def trade(self) -> None:
        log.info("Trade loop starting in 5s...")
        await asyncio.sleep(5)
        tick = 0
        while self._running:
            try:
                self._update_rate_expectation()
                if ENABLE_STRAT_A:
                    await self.strat_market_make_A()
                if ENABLE_STRAT_ETF:
                    await self.strat_etf_nav_arb()
                if ENABLE_STRAT_PARITY:
                    await self.strat_parity_B()

                # Snapshot every ~5 seconds (every 10 loop iterations at 0.5s each)
                if tick % 10 == 0:
                    mids = {sym: self._mid(sym) for sym in self.order_books.keys()}
                    log_event("snapshot",
                              positions=dict(self.positions),
                              mids={k: v for k, v in mids.items() if v is not None},
                              rate_exp_bps=self.fv.rate_expectation_bps,
                              fv_A=self.fv.A,
                              rejects_total=self._reject_count)
                tick += 1
            except Exception:
                log.error("trade loop error:\n%s", traceback.format_exc())
            await asyncio.sleep(LOOP_INTERVAL)

    async def start(self) -> None:
        asyncio.create_task(self.trade())
        await self.connect()


# ---------------------------------------------------------------------------

async def main() -> None:
    bot = Case1Bot(SERVER, USERNAME, PASSWORD)
    try:
        await bot.start()
    except KeyboardInterrupt:
        log.info("Shutting down.")
    except Exception:
        log.error("Fatal:\n%s", traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())