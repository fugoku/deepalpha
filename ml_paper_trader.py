"""Historical-indicator learning and persistent paper-only spot trading."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import logging
import os
import pickle
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from catalyst_monitor import (
    fetch_market_catalysts,
    recent_high_impact_events,
)

import config
from canadian_arbitrage import (
    CANADIAN_VENUES,
    fetch_canadian_spot_snapshots,
    validate_fee_rate,
)
from whale_flow import fetch_canadian_whale_flows

logger = logging.getLogger("canadian-paper-trader")
HORIZON = 4
MIN_TRAINING_SAMPLES = 120
MODEL_RETRAIN_SECONDS = 24 * 60 * 60
DEFAULT_POLL_SECONDS = 300
DEFAULT_CANDLE_COUNT = 720
PAPER_MAX_HOLD_SECONDS = 24 * 60 * 60
PAPER_ENTRY_PROBABILITY = 0.60
PAPER_EXIT_PROBABILITY = 0.48
PAPER_ALLOCATION_FRACTION = 0.05
PAPER_MAX_POSITIONS = 3
SLIPPAGE_BUFFER = 0.001
GPU_TRAINING_EPOCHS = 60
GPU_TRAINING_BATCH_SIZE = 256
ESTIMATED_BYTES_PER_CANDLE = 16 * 1024
CATALYST_REFRESH_SECONDS = 30 * 60
WHALE_REFRESH_SECONDS = 15 * 60
WHALE_ALIGNMENT_THRESHOLD = 0.65
WHALE_MIN_CONFIRMATIONS = 2

INDICATOR_NAMES = (
    "return_1", "return_4", "return_12", "return_24",
    "ema_12_26_pct", "rsi_14", "macd_hist_pct", "atr_14_pct",
    "bollinger_percent_b", "bollinger_width_pct", "stochastic_14",
    "volume_ratio_20", "vwap_distance_pct", "obv_slope_20",
    "sma_10_20_pct", "sma_20_50_pct", "sma_50_200_pct",
    "ema_9_21_pct", "ema_21_50_pct", "rsi_7", "rsi_21",
    "roc_5", "roc_10", "roc_20", "macd_line_pct", "macd_signal_pct",
    "realized_volatility_10", "realized_volatility_20",
    "stochastic_d_3", "williams_r_14",
    "adx_14", "plus_di_14", "minus_di_14", "cci_20", "cmf_20",
    "mfi_14", "donchian_position_20", "donchian_position_50",
    "candle_body_pct", "close_location_value", "volume_ratio_5",
    "volume_price_trend_20", "high_low_range_pct", "overnight_gap_pct",
)


def whale_direction_confirmation(
    long_probability: float,
    short_probability: float,
    flow: dict[str, Any] | None,
) -> tuple[str | None, str]:
    """Return a model side only when it is not strongly opposed by observed whale flow."""
    side = "long" if long_probability >= short_probability else "short"
    if not flow or flow.get("whale_count", 0) < WHALE_MIN_CONFIRMATIONS:
        return side, "No repeated large-trade flow; no whale confirmation applied."
    ratio = float(flow.get("flow_ratio", 0))
    if abs(ratio) < WHALE_ALIGNMENT_THRESHOLD:
        return side, f"Large-trade flow is mixed (imbalance {ratio:+.2f})."
    flow_side = "long" if ratio > 0 else "short"
    if side != flow_side:
        return None, (
            f"Entry blocked: model favors {side}, but repeated large trades show "
            f"strong {flow_side} pressure ({ratio:+.2f})."
        )
    return side, f"Model direction confirmed by large-trade flow ({ratio:+.2f})."


def build_indicator_matrix(candles: list[Any]) -> np.ndarray:
    """Create causal price/volume features using only candles available at each row."""
    if not candles:
        return np.empty((0, len(INDICATOR_NAMES)))
    if isinstance(candles[0], dict):
        open_ = np.array([float(c.get("o", 0)) for c in candles])
        high = np.array([float(c.get("h", 0)) for c in candles])
        low = np.array([float(c.get("l", 0)) for c in candles])
        close = np.array([float(c.get("c", 0)) for c in candles])
        volume = np.array([float(c.get("v", 0)) for c in candles])
    else:
        open_ = np.array([float(c[1]) for c in candles])
        high = np.array([float(c[2]) for c in candles])
        low = np.array([float(c[3]) for c in candles])
        close = np.array([float(c[4]) for c in candles])
        volume = np.array([float(c[5]) for c in candles])

    n = len(close)
    if any(len(values) != n for values in (open_, high, low, volume)):
        raise ValueError("OHLCV fields must have matching lengths.")
    if np.any(close <= 0) or np.any(high <= 0) or np.any(low <= 0):
        raise ValueError("OHLC prices must be positive.")

    def ema(values: np.ndarray, period: int) -> np.ndarray:
        result = np.empty_like(values)
        result[0] = values[0]
        alpha = 2 / (period + 1)
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
        return result

    def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(values), np.nan)
        if len(values) >= window:
            result[window - 1:] = np.convolve(
                values, np.ones(window) / window, mode="valid"
            )
        return result

    def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
        result = np.full(len(values), np.nan)
        if len(values) >= window:
            result[window - 1:] = np.array([
                np.std(values[i - window + 1:i + 1], ddof=0)
                for i in range(window - 1, len(values))
            ])
        return result

    def rolling_extreme(values: np.ndarray, window: int, maximum: bool) -> np.ndarray:
        result = np.full(len(values), np.nan)
        if len(values) >= window:
            reducer = np.max if maximum else np.min
            result[window - 1:] = np.array([
                reducer(values[i - window + 1:i + 1])
                for i in range(window - 1, len(values))
            ])
        return result

    def rsi_for_period(period: int) -> np.ndarray:
        period_gain = rolling_mean(gains, period)
        period_loss = rolling_mean(losses, period)
        rs = np.divide(
            period_gain,
            period_loss,
            out=np.full(n, np.nan),
            where=period_loss > 0,
        )
        result = 100 - 100 / (1 + rs)
        result = np.where((period_loss == 0) & (period_gain > 0), 100, result)
        return np.where((period_loss == 0) & (period_gain == 0), 50, result)

    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1]
    returns_pct = np.diff(close, prepend=close[0])
    returns_pct = returns_pct / close
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26
    macd_signal = ema(macd_line, 9)
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)

    gains = np.maximum(np.diff(close, prepend=close[0]), 0)
    losses = np.maximum(-np.diff(close, prepend=close[0]), 0)
    avg_gain = rolling_mean(gains, 14)
    avg_loss = rolling_mean(losses, 14)
    relative_strength = np.divide(
        avg_gain,
        avg_loss,
        out=np.full(n, np.nan),
        where=avg_loss > 0,
    )
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = np.where((avg_loss == 0) & (avg_gain > 0), 100, rsi)
    rsi = np.where((avg_loss == 0) & (avg_gain == 0), 50, rsi)

    rsi7 = rsi_for_period(7)
    rsi21 = rsi_for_period(21)
    previous_close = np.roll(close, 1)
    previous_close[0] = close[0]
    true_range = np.maximum(
        high - low,
        np.maximum(np.abs(high - previous_close), np.abs(low - previous_close)),
    )
    atr = rolling_mean(true_range, 14)

    mean20 = rolling_mean(close, 20)
    std20 = rolling_std(close, 20)
    upper = mean20 + 2 * std20
    lower = mean20 - 2 * std20
    band_width = upper - lower
    percent_b = (close - lower) / np.where(band_width > 0, band_width, np.nan)

    high14 = rolling_extreme(high, 14, True)
    low14 = rolling_extreme(low, 14, False)
    stochastic = 100 * (close - low14) / np.where(high14 > low14, high14 - low14, np.nan)
    stochastic_d = rolling_mean(stochastic, 3)
    williams_r = -100 * (high14 - close) / np.where(high14 > low14, high14 - low14, np.nan)

    vol_mean20 = rolling_mean(volume, 20)
    vol_mean5 = rolling_mean(volume, 5)
    typical = (high + low + close) / 3
    vwap_numerator = rolling_mean(typical * volume, 20)
    vwap_denominator = rolling_mean(volume, 20)
    vwap = vwap_numerator / np.where(vwap_denominator > 0, vwap_denominator, np.nan)

    signed_volume = volume * np.sign(returns)
    obv = np.cumsum(signed_volume)
    obv_mean = rolling_mean(obv, 20)
    obv_slope = (obv - obv_mean) / np.where(vol_mean20 > 0, vol_mean20 * 20, np.nan)
    volume_price_trend = np.cumsum(returns * volume)
    vpt_mean = rolling_mean(volume_price_trend, 20)
    vpt_slope = (volume_price_trend - vpt_mean) / np.where(
        vol_mean20 > 0, vol_mean20 * 20, np.nan
    )

    sma10 = rolling_mean(close, 10)
    sma50 = rolling_mean(close, 50)
    sma200 = rolling_mean(close, 200)
    volatility10 = rolling_std(returns, 10) * np.sqrt(10)
    volatility20 = rolling_std(returns, 20) * np.sqrt(20)

    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    smooth_tr = rolling_mean(true_range, 14)
    plus_di = 100 * rolling_mean(plus_dm, 14) / np.where(smooth_tr > 0, smooth_tr, np.nan)
    minus_di = 100 * rolling_mean(minus_dm, 14) / np.where(smooth_tr > 0, smooth_tr, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di > 0, plus_di + minus_di, np.nan)
    adx = rolling_mean(dx, 14)

    typical_mean20 = rolling_mean(typical, 20)
    mean_deviation20 = np.full(n, np.nan)
    for i in range(19, n):
        mean_deviation20[i] = np.mean(np.abs(typical[i - 19:i + 1] - typical_mean20[i]))
    cci = (typical - typical_mean20) / np.where(mean_deviation20 > 0, 0.015 * mean_deviation20, np.nan)

    money_flow = typical * volume
    money_flow_direction = np.sign(np.diff(typical, prepend=typical[0]))
    positive_flow = rolling_mean(np.where(money_flow_direction > 0, money_flow, 0), 14) * 14
    negative_flow = rolling_mean(np.where(money_flow_direction < 0, money_flow, 0), 14) * 14
    money_ratio = positive_flow / np.where(negative_flow > 0, negative_flow, np.nan)
    mfi = 100 - 100 / (1 + money_ratio)
    mfi = np.where((negative_flow == 0) & (positive_flow > 0), 100, mfi)
    mfi = np.where((negative_flow == 0) & (positive_flow == 0), 50, mfi)

    money_flow_multiplier = np.divide(
        (close - low) - (high - close),
        high - low,
        out=np.zeros(n),
        where=(high - low) > 0,
    )
    cmf_denominator = rolling_mean(volume, 20)
    cmf = rolling_mean(money_flow_multiplier * volume, 20) / np.where(
        cmf_denominator > 0, cmf_denominator, np.nan
    )
    high20 = rolling_extreme(high, 20, True)
    low20 = rolling_extreme(low, 20, False)
    high50 = rolling_extreme(high, 50, True)
    low50 = rolling_extreme(low, 50, False)
    previous_open = np.roll(open_, 1)
    previous_open[0] = open_[0]

    def pct_change(period: int) -> np.ndarray:
        result = np.full(n, np.nan)
        if n > period:
            result[period:] = (close[period:] - close[:-period]) / close[:-period]
        return result

    matrix = np.column_stack((
        returns,
        pct_change(4),
        pct_change(12),
        pct_change(24),
        (ema12 - ema26) / close,
        rsi / 100,
        (macd_line - macd_signal) / close,
        atr / close,
        percent_b,
        band_width / close,
        stochastic / 100,
        volume / np.where(vol_mean20 > 0, vol_mean20, np.nan),
        (close - vwap) / vwap,
        obv_slope,
        (close - sma10) / sma10,
        (mean20 - sma50) / np.where(sma50 > 0, sma50, np.nan),
        (sma50 - sma200) / np.where(sma200 > 0, sma200, np.nan),
        (ema9 - ema21) / close,
        (ema21 - ema50) / close,
        rsi7 / 100,
        rsi21 / 100,
        pct_change(5),
        pct_change(10),
        pct_change(20),
        macd_line / close,
        macd_signal / close,
        volatility10,
        volatility20,
        stochastic_d / 100,
        williams_r / 100,
        adx / 100,
        plus_di / 100,
        minus_di / 100,
        cci / 200,
        cmf,
        mfi / 100,
        (close - low20) / np.where(high20 > low20, high20 - low20, np.nan),
        (close - low50) / np.where(high50 > low50, high50 - low50, np.nan),
        (close - open_) / open_,
        money_flow_multiplier,
        volume / np.where(vol_mean5 > 0, vol_mean5, np.nan),
        vpt_slope,
        (high - low) / close,
        (open_ - previous_open) / previous_open,
    ))
    if matrix.shape[1] != len(INDICATOR_NAMES):
        raise RuntimeError(
            f"Indicator definition mismatch: {matrix.shape[1]} values for "
            f"{len(INDICATOR_NAMES)} names."
        )
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_historical_samples(
    candles_by_asset: dict[str, list[Any]],
    fee_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Label four-hour upward/downward moves only when they beat estimated round-trip costs."""
    fee_rate = validate_fee_rate(fee_rate)
    samples = []
    cost_threshold = 2 * (fee_rate + SLIPPAGE_BUFFER)
    for candles in candles_by_asset.values():
        features = build_indicator_matrix(candles)
        close = np.array([
            float(c["c"] if isinstance(c, dict) else c[4]) for c in candles
        ])
        last_labeled_index = len(close) - HORIZON
        if last_labeled_index <= 30:
            continue
        future_returns = (close[HORIZON:] - close[:-HORIZON]) / close[:-HORIZON]
        indices = np.arange(30, last_labeled_index)
        long_labels = (future_returns[indices] > cost_threshold).astype(int)
        short_labels = (future_returns[indices] < -cost_threshold).astype(int)
        for index, long_label, short_label in zip(indices, long_labels, short_labels):
            candle = candles[index]
            timestamp = candle[0] if not isinstance(candle, dict) else candle.get("t", index)
            samples.append(
                (int(timestamp), features[index], int(long_label), int(short_label))
            )
    if not samples:
        empty = np.empty(0, dtype=int)
        return np.empty((0, len(INDICATOR_NAMES))), empty, empty, empty
    samples.sort(key=lambda row: row[0])
    return (
        np.asarray([row[1] for row in samples]),
        np.asarray([row[2] for row in samples], dtype=int),
        np.asarray([row[3] for row in samples], dtype=int),
        np.asarray([row[0] for row in samples], dtype=np.int64),
    )


def chronological_split(
    timestamps: np.ndarray,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by timestamp and purge training labels whose future window reaches test data."""
    timestamps = np.asarray(timestamps)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("At least two one-dimensional timestamps are required.")
    unique_timestamps = np.unique(timestamps)
    if len(unique_timestamps) < 2:
        raise ValueError("At least two distinct timestamps are required.")
    cutoff_index = int(len(unique_timestamps) * 0.8)
    cutoff_index = min(max(cutoff_index, 1), len(unique_timestamps) - 1)
    cutoff = unique_timestamps[cutoff_index]
    deltas = np.diff(unique_timestamps)
    typical_interval = np.median(deltas[deltas > 0])
    embargo = typical_interval * horizon
    train_mask = timestamps < cutoff - embargo
    test_mask = timestamps >= cutoff
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Not enough distinct historical timestamps after purging the split.")
    return train_mask, test_mask


def _positive_class_weight(labels: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Training labels must contain both classes.")
    return negatives / positives


class PaperLearningEngine:
    """SQLite-backed long/short paper account with historical and outcome retraining."""

    def __init__(
        self,
        database_path: str,
        model_path: str,
        *,
        starting_capital_cad: float = 10_000,
        allocation_fraction: float = PAPER_ALLOCATION_FRACTION,
        max_positions: int = PAPER_MAX_POSITIONS,
        device: str = "auto",
    ):
        if starting_capital_cad <= 0 or not 0 < allocation_fraction <= 1:
            raise ValueError("Paper capital must be positive and allocation must be in (0, 1].")
        self.database_path = database_path
        self.model_path = model_path
        self.starting_capital_cad = starting_capital_cad
        self.allocation_fraction = allocation_fraction
        self.max_positions = max_positions
        self.device = _resolve_device(device)
        os.makedirs(os.path.dirname(os.path.abspath(database_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        self.models = {"long": None, "short": None}
        self._gpu_estimators = {}
        self._initialize_database()
        if os.path.exists(model_path):
            with open(model_path, "rb") as model_file:
                saved = pickle.load(model_file)
            if isinstance(saved, dict) and "backend" in saved:
                if saved["backend"] != self.device:
                    raise ValueError(
                        f"Saved model uses {saved['backend']} training. "
                        f"Start with --device {saved['backend']} to use it."
                    )
                self.models = saved["models"]
            else:
                self.models = saved

    def _connect(self):
        return sqlite3.connect(self.database_path)

    def _initialize_database(self):
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY,
                    asset TEXT NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    capital_cad REAL NOT NULL,
                    entry_fee_cad REAL NOT NULL,
                    exit_fee_cad REAL,
                    net_pnl_cad REAL,
                    entry_features TEXT NOT NULL,
                    entry_probability REAL NOT NULL,
                    entry_exchange TEXT NOT NULL,
                    exit_exchange TEXT
                )
            """)
            columns = {
                row[1] for row in db.execute("PRAGMA table_info(paper_trades)").fetchall()
            }
            if "side" not in columns:
                db.execute(
                    "ALTER TABLE paper_trades ADD COLUMN side TEXT NOT NULL DEFAULT 'long'"
                )
            db.execute("""
                CREATE TABLE IF NOT EXISTS market_catalysts (
                    url TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    impact TEXT NOT NULL,
                    tone REAL NOT NULL,
                    published_at TEXT NOT NULL,
                    stored_at TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS whale_flow_snapshots (
                    asset TEXT PRIMARY KEY,
                    buy_notional_cad REAL NOT NULL,
                    sell_notional_cad REAL NOT NULL,
                    net_flow_cad REAL NOT NULL,
                    flow_ratio REAL NOT NULL,
                    whale_count INTEGER NOT NULL,
                    venues TEXT NOT NULL,
                    observed_at TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS whale_prints (
                    id INTEGER PRIMARY KEY,
                    exchange_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    notional_cad REAL NOT NULL,
                    threshold_cad REAL NOT NULL,
                    flow_destination TEXT NOT NULL,
                    UNIQUE(exchange_id, symbol, timestamp_ms, side, notional_cad)
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS service_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

    def update_service_metadata(self, values: dict[str, Any]) -> None:
        with self._connect() as db:
            db.executemany(
                "INSERT INTO service_metadata (key,value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [(key, json.dumps(value)) for key, value in values.items()],
            )

    def store_catalysts(self, events: list[dict[str, Any]]) -> None:
        stored_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.executemany(
                "INSERT OR IGNORE INTO market_catalysts "
                "(url,title,source,category,impact,tone,published_at,stored_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        event["url"], event["title"], event["source"],
                        event["category"], event["impact"], event["tone"],
                        event["published_at"], stored_at,
                    )
                    for event in events
                ],
            )
            db.execute(
                "DELETE FROM market_catalysts WHERE published_at < ?",
                ((
                    datetime.now(timezone.utc) - timedelta(days=3)
                ).isoformat(),),
            )

    def recent_catalysts(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT url,title,source,category,impact,tone,published_at "
                "FROM market_catalysts ORDER BY published_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def store_whale_observations(
        self,
        flows: dict[str, dict[str, Any]],
        prints: list[dict[str, Any]],
    ) -> None:
        observed_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.executemany(
                "INSERT INTO whale_flow_snapshots "
                "(asset,buy_notional_cad,sell_notional_cad,net_flow_cad,flow_ratio,"
                "whale_count,venues,observed_at) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(asset) DO UPDATE SET "
                "buy_notional_cad=excluded.buy_notional_cad,"
                "sell_notional_cad=excluded.sell_notional_cad,"
                "net_flow_cad=excluded.net_flow_cad,flow_ratio=excluded.flow_ratio,"
                "whale_count=excluded.whale_count,venues=excluded.venues,"
                "observed_at=excluded.observed_at",
                [
                    (
                        asset, data["buy_notional_cad"], data["sell_notional_cad"],
                        data["net_flow_cad"], data["flow_ratio"], data["whale_count"],
                        json.dumps(data["venues"]), observed_at,
                    )
                    for asset, data in flows.items()
                ],
            )
            db.executemany(
                "INSERT OR IGNORE INTO whale_prints "
                "(exchange_id,symbol,timestamp_ms,side,notional_cad,threshold_cad,"
                "flow_destination) VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        trade["exchange"], trade["symbol"], trade["timestamp_ms"],
                        trade["side"], trade["notional_cad"], trade["threshold_cad"],
                        trade["flow_destination"],
                    )
                    for trade in prints
                ],
            )
            db.execute("""
                DELETE FROM whale_prints
                WHERE timestamp_ms < ?
            """, (int(datetime.now(timezone.utc).timestamp() * 1000) - 7 * 24 * 3600 * 1000,))

    def latest_whale_observations(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            flows = db.execute(
                "SELECT asset,buy_notional_cad,sell_notional_cad,net_flow_cad,flow_ratio,"
                "whale_count,venues,observed_at FROM whale_flow_snapshots "
                "ORDER BY ABS(net_flow_cad) DESC"
            ).fetchall()
            prints = db.execute(
                "SELECT exchange_id,symbol,timestamp_ms,side,notional_cad,"
                "threshold_cad,flow_destination FROM whale_prints "
                "ORDER BY timestamp_ms DESC LIMIT 50"
            ).fetchall()
        return [dict(row) for row in flows], [dict(row) for row in prints]

    def train(
        self,
        historical_x: np.ndarray,
        long_y: np.ndarray,
        short_y: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> dict[str, float] | None:
        """Evaluate chronologically, then fit on history plus completed paper-trade labels."""
        if len(long_y) < MIN_TRAINING_SAMPLES:
            logger.warning(
                "Not enough historical samples to train (need at least %d).",
                MIN_TRAINING_SAMPLES,
            )
            return None

        if timestamps is not None and len(timestamps) != len(long_y):
            raise ValueError("Historical timestamps must align with feature and label rows.")
        if timestamps is not None:
            try:
                train_mask, test_mask = chronological_split(timestamps)
            except ValueError as exc:
                logger.warning("Cannot split historical data chronologically: %s", exc)
                return None
            split = int(train_mask.sum())
        else:
            split = int(len(long_y) * 0.8)
            train_mask = np.arange(len(long_y)) < split
            test_mask = ~train_mask
        if split < 1 or split >= len(long_y) or not test_mask.any():
            logger.warning("Historical data cannot be split for chronological evaluation.")
            return None

        accuracies = {}
        trained_models = {}
        for side, labels in (("long", long_y), ("short", short_y)):
            if len(np.unique(labels[train_mask])) < 2:
                logger.warning("Skipping %s model: historical training window has one class.", side)
                trained_models[side] = None
                continue
            paper_x, paper_y = self._paper_samples(side)
            historical_test_x = historical_x[test_mask]
            historical_test_y = labels[test_mask]
            fit_x = historical_x[train_mask]
            fit_y = labels[train_mask]
            if self.device == "cuda":
                evaluator_state, evaluator_mean, evaluator_scale = _fit_cuda_classifier(
                    fit_x, fit_y
                )
                test_probabilities = _predict_cuda(
                    evaluator_state, evaluator_mean, evaluator_scale, historical_test_x
                )
                accuracies[side] = float(
                    np.mean((test_probabilities >= 0.5) == historical_test_y)
                )
            else:
                evaluator = make_pipeline(
                    StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
                )
                evaluator.fit(fit_x, fit_y)
                accuracies[side] = float(evaluator.score(historical_test_x, historical_test_y))

            if len(paper_y):
                train_x = np.vstack((historical_x, paper_x))
                train_y = np.concatenate((labels, paper_y))
            else:
                train_x, train_y = historical_x, labels
            if len(np.unique(train_y)) < 2:
                trained_models[side] = None
                continue
            if self.device == "cuda":
                state, mean, scale = _fit_cuda_classifier(train_x, train_y)
                trained_models[side] = {
                    "state": state,
                    "mean": mean,
                    "scale": scale,
                }
            else:
                model = make_pipeline(
                    StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
                )
                model.fit(train_x, train_y)
                trained_models[side] = model

        if not any(trained_models.values()):
            logger.warning("No direction model could be trained from the current history.")
            return None
        self.models = trained_models
        self._gpu_estimators = {}
        with open(self.model_path, "wb") as model_file:
            pickle.dump({"backend": self.device, "models": self.models}, model_file)
        self.update_service_metadata({
            "model_trained_at": datetime.now(timezone.utc).isoformat(),
            "model_backend": self.device,
            "training_samples": int(len(long_y)),
            "long_accuracy": accuracies.get("long"),
            "short_accuracy": accuracies.get("short"),
            "feature_count": len(INDICATOR_NAMES),
        })
        logger.info("Direction models retrained; chronological holdout accuracy: %s.",
                    ", ".join(f"{side} {score:.1%}" for side, score in accuracies.items()))
        return accuracies

    def _paper_samples(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT entry_features, net_pnl_cad FROM paper_trades "
                "WHERE status = 'closed' AND net_pnl_cad IS NOT NULL AND side = ?",
                (side,),
            ).fetchall()
        if not rows:
            return np.empty((0, len(INDICATOR_NAMES))), np.empty(0, dtype=int)
        return (
            np.asarray([json.loads(row[0]) for row in rows], dtype=float),
            np.asarray([int(row[1] > 0) for row in rows], dtype=int),
        )

    def probabilities(
        self, candles_by_asset: dict[str, list[Any]]
    ) -> dict[str, tuple[float, float]]:
        predictions = {}
        for asset, candles in candles_by_asset.items():
            if len(candles) < 30:
                continue
            completed_candles = candles[:-1]
            if len(completed_candles) < 30:
                continue
            features = build_indicator_matrix(completed_candles)[-1:]
            side_probabilities = {}
            for side, model in self.models.items():
                if model is None:
                    continue
                if self.device == "cuda":
                    estimator = self._gpu_estimators.get(side)
                    if estimator is None:
                        import torch

                        estimator = _build_torch_network(len(INDICATOR_NAMES)).to("cuda:0")
                        estimator.load_state_dict(model["state"])
                        estimator.eval()
                        self._gpu_estimators[side] = estimator
                    scaled = (features - model["mean"]) / model["scale"]
                    with torch.inference_mode():
                        side_probability = torch.sigmoid(
                            estimator(torch.as_tensor(scaled, dtype=torch.float32, device="cuda:0"))
                        ).item()
                    side_probabilities[side] = float(side_probability)
                else:
                    side_probabilities[side] = float(model.predict_proba(features)[0, 1])
            predictions[asset] = (
                side_probabilities.get("long", 0.0),
                side_probabilities.get("short", 0.0),
            )
        return predictions

    def update_paper_positions(
        self,
        snapshots: dict[str, dict[str, Any]],
        candles_by_asset: dict[str, list[Any]],
        probabilities: dict[str, tuple[float, float]],
        *,
        whale_flows: dict[str, dict[str, Any]] | None = None,
        pause_new_entries: bool = False,
    ) -> None:
        """Open/close simulated CAD spot long/short positions using public quotes only."""
        now = datetime.now(timezone.utc).isoformat()
        best_quotes = {}
        for exchange_id, data in snapshots.items():
            for symbol, ticker in data.get("tickers", {}).items():
                market = data.get("markets", {}).get(symbol, {})
                if (
                    not market.get("spot", False)
                    or market.get("quote") != "CAD"
                    or market.get("base") not in config.ARBITRAGE_ASSETS
                ):
                    continue
                bid, ask = ticker.get("bid"), ticker.get("ask")
                if bid is None or ask is None or bid <= 0 or ask < bid:
                    continue
                current = best_quotes.setdefault(market["base"], {})
                quote = {"symbol": symbol, "bid": float(bid), "ask": float(ask),
                         "exchange": exchange_id,
                         "fee": validate_fee_rate(data["fee_rate"])}
                if "best_bid" not in current or quote["bid"] > current["best_bid"]["bid"]:
                    current["best_bid"] = quote
                if "best_ask" not in current or quote["ask"] < current["best_ask"]["ask"]:
                    current["best_ask"] = quote

        with self._connect() as db:
            open_rows = db.execute(
                "SELECT id, asset, side, entry_time, capital_cad, quantity, entry_fee_cad, "
                "entry_features, entry_probability, entry_price "
                "FROM paper_trades WHERE status = 'open'"
            ).fetchall()
            realized = db.execute(
                "SELECT COALESCE(SUM(net_pnl_cad), 0) FROM paper_trades WHERE status = 'closed'"
            ).fetchone()[0]
            occupied = sum(row[4] for row in open_rows)
            equity = self.starting_capital_cad + realized

            for row in open_rows:
                trade_id, asset, side, entry_time, capital, quantity, entry_fee, _, _, entry_price = row
                quotes = best_quotes.get(asset)
                if not quotes:
                    continue
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(entry_time)).total_seconds()
                long_probability, short_probability = probabilities.get(asset, (0.0, 0.0))
                probability = long_probability if side == "long" else short_probability
                if probability > PAPER_EXIT_PROBABILITY and age < PAPER_MAX_HOLD_SECONDS:
                    continue
                exit_quote = quotes["best_bid"] if side == "long" else quotes["best_ask"]
                exit_price = exit_quote["bid"] if side == "long" else exit_quote["ask"]
                exit_fee = quantity * exit_price * exit_quote["fee"]
                if side == "long":
                    net_pnl = quantity * exit_price - capital - entry_fee - exit_fee
                else:
                    net_pnl = quantity * (entry_price - exit_price) - entry_fee - exit_fee
                db.execute(
                    "UPDATE paper_trades SET status='closed', exit_time=?, exit_price=?, "
                    "exit_fee_cad=?, net_pnl_cad=?, exit_exchange=? WHERE id=?",
                    (now, exit_price, exit_fee, net_pnl, exit_quote["exchange"], trade_id),
                )
                occupied -= capital
                logger.info("Closed paper %s: net PnL CAD %.2f after fees.", asset, net_pnl)

            open_assets = {row[1] for row in open_rows}
            for asset, (long_probability, short_probability) in probabilities.items():
                if (
                    pause_new_entries
                    or
                    max(long_probability, short_probability) < PAPER_ENTRY_PROBABILITY
                    or asset in open_assets
                    or asset not in best_quotes
                ):
                    continue
                open_count = db.execute(
                    "SELECT COUNT(*) FROM paper_trades WHERE status='open'"
                ).fetchone()[0]
                if open_count >= self.max_positions:
                    break
                side, decision_reason = whale_direction_confirmation(
                    long_probability,
                    short_probability,
                    (whale_flows or {}).get(asset),
                )
                if side is None:
                    logger.info("Skipped %s: %s", asset, decision_reason)
                    continue
                if (
                    (side == "long" and long_probability < PAPER_ENTRY_PROBABILITY)
                    or (side == "short" and short_probability < PAPER_ENTRY_PROBABILITY)
                ):
                    continue
                entry_quote = (
                    best_quotes[asset]["best_ask"] if side == "long"
                    else best_quotes[asset]["best_bid"]
                )
                entry_price = entry_quote["ask"] if side == "long" else entry_quote["bid"]
                probability = long_probability if side == "long" else short_probability
                capital = min(equity * self.allocation_fraction, equity - occupied)
                if capital <= 0:
                    break
                execution_price = entry_price * (
                    1 + SLIPPAGE_BUFFER if side == "long" else 1 - SLIPPAGE_BUFFER
                )
                quantity = capital / execution_price
                entry_fee = capital * entry_quote["fee"]
                completed_candles = candles_by_asset[asset][:-1]
                if len(completed_candles) < 30:
                    continue
                features = build_indicator_matrix(completed_candles)[-1].tolist()
                db.execute(
                    "INSERT INTO paper_trades "
                    "(asset,side,status,entry_time,entry_price,quantity,capital_cad,entry_fee_cad,"
                    "entry_features,entry_probability,entry_exchange) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (asset, side, "open", now, execution_price, quantity, capital, entry_fee, json.dumps(features),
                     probability, entry_quote["exchange"]),
                )
                open_assets.add(asset)
                occupied += capital
                logger.info(
                    "Opened paper %s %s at CAD %.6g; model probability %.1f%%. %s",
                    side, asset, entry_price, probability * 100, decision_reason,
                )


def _download_history(exchange_id: str, candle_count: int) -> tuple[dict[str, list[Any]], float]:
    import ccxt

    client = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 20000})
    client.load_markets()
    history = {}
    for asset in sorted(config.ARBITRAGE_ASSETS):
        symbol = f"{asset}/CAD"
        market = client.markets.get(symbol)
        if not market or not market.get("spot") or market.get("active") is False:
            continue
        try:
            candles = client.fetch_ohlcv(symbol, timeframe="1h", limit=candle_count)
            if len(candles) >= 100:
                history[asset] = candles
        except Exception as exc:
            logger.warning("Historical candles unavailable for %s on %s: %s", symbol, exchange_id, exc)
    return history, config.ARBITRAGE_TAKER_FEES.get(exchange_id, 0.005)


def _resolve_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("ML device must be 'auto', 'cpu', or 'cuda'.")
    if device in {"auto", "cpu"}:
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CUDA training requires a CUDA-enabled PyTorch install. "
            "Install the build matching your NVIDIA driver, then retry."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested, but CUDA is not available.")
    _require_cuda_device(torch.cuda.device_count())
    return "cuda"


def _require_cuda_device(device_count: int) -> None:
    if device_count < 1:
        raise RuntimeError(
            "CUDA training was requested, but no CUDA devices are visible."
        )


def available_memory_bytes() -> int:
    """Return OS-reported available RAM, without adding a runtime dependency."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Windows could not report available physical memory.")
        return int(status.ullAvailPhys)
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError) as exc:
        raise OSError("The operating system could not report available physical memory.") from exc


def resolve_memory_budget(
    *,
    max_ram_gb: float = 0,
    fraction: float = config.PAPER_MEMORY_FRACTION,
    available_bytes: int | None = None,
) -> tuple[int, int]:
    """Return (available RAM, task budget), preserving the configured OS headroom."""
    if not 0 < fraction <= 0.8:
        raise ValueError("PAPER_MEMORY_FRACTION must be greater than 0 and at most 0.8.")
    if max_ram_gb < 0:
        raise ValueError("Maximum RAM must not be negative.")
    available = available_memory_bytes() if available_bytes is None else available_bytes
    budget = int(available * fraction)
    if max_ram_gb:
        budget = min(budget, int(max_ram_gb * 1024**3))
    if budget <= 0:
        raise RuntimeError("The paper trader has no usable RAM budget.")
    return available, budget


def cap_candle_history(
    requested_candles: int,
    budget_bytes: int,
    asset_count: int,
) -> int:
    """Bound historical downloads using a conservative per-candle working-set estimate."""
    if requested_candles < 100:
        raise ValueError("At least 100 candles are required for historical training.")
    if budget_bytes <= 0 or asset_count <= 0:
        raise ValueError("A positive memory budget and asset count are required.")
    capacity = budget_bytes // (asset_count * ESTIMATED_BYTES_PER_CANDLE)
    if capacity < 100:
        raise RuntimeError(
            "The available RAM budget is too small to retain 100 hourly candles "
            "for each configured asset."
        )
    return min(requested_candles, int(capacity))


def _build_torch_network(input_count: int):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(input_count, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )


def _fit_cuda_classifier(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Train one tabular classifier across one or two visible CUDA devices."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    scaler = StandardScaler().fit(features)
    scaled_features = scaler.transform(features).astype(np.float32)
    torch.manual_seed(42)
    dataset = TensorDataset(
        torch.from_numpy(scaled_features),
        torch.as_tensor(labels, dtype=torch.float32).reshape(-1, 1),
    )
    loader = DataLoader(
        dataset,
        batch_size=GPU_TRAINING_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    device_ids = list(range(min(2, torch.cuda.device_count())))
    network = _build_torch_network(features.shape[1]).to("cuda:0")
    model = (
        torch.nn.DataParallel(network, device_ids=device_ids)
        if len(device_ids) == 2
        else network
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    positive_weight = _positive_class_weight(labels)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device="cuda:0")
    )
    model.train()
    for _ in range(GPU_TRAINING_EPOCHS):
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to("cuda:0", non_blocking=True)
            batch_labels = batch_labels.to("cuda:0", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

    trained_network = model.module if isinstance(model, torch.nn.DataParallel) else model
    state = {
        key: value.detach().cpu().clone()
        for key, value in trained_network.state_dict().items()
    }
    return state, scaler.mean_.copy(), scaler.scale_.copy()


def _predict_cuda(
    state: dict[str, Any],
    mean: np.ndarray,
    scale: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    import torch

    model = _build_torch_network(features.shape[1]).to("cuda:0")
    model.load_state_dict(state)
    model.eval()
    scaled = (features - mean) / scale
    with torch.inference_mode():
        return torch.sigmoid(
            model(torch.as_tensor(scaled, dtype=torch.float32, device="cuda:0"))
        ).cpu().numpy().reshape(-1)


def run_forever(
    *,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    candle_count: int = DEFAULT_CANDLE_COUNT,
    database_path: str | None = None,
    model_path: str | None = None,
    starting_capital_cad: float = 10_000,
    device: str = "auto",
    max_ram_gb: float = config.PAPER_MAX_RAM_GB,
) -> None:
    if poll_seconds < 30:
        raise ValueError("Polling interval must be at least 30 seconds.")
    available, ram_budget = resolve_memory_budget(max_ram_gb=max_ram_gb)
    candle_count = cap_candle_history(
        candle_count, ram_budget, len(config.ARBITRAGE_ASSETS)
    )
    logger.info(
        "RAM budget %.2f GB from %.2f GB available (%.0f%% cap); history limited to %d "
        "hourly candles per asset.",
        ram_budget / 1024**3,
        available / 1024**3,
        ram_budget / available * 100,
        candle_count,
    )
    data_dir = config.DATA_DIR
    database_path = database_path or os.path.join(data_dir, "canadian_paper_trades.sqlite3")
    model_path = model_path or os.path.join(data_dir, "canadian_paper_model.pkl")
    engine = PaperLearningEngine(
        database_path,
        model_path,
        starting_capital_cad=starting_capital_cad,
        device=device,
    )
    venue_ids = tuple(name for name in config.CANADIAN_ARBITRAGE_EXCHANGES
                      if name in CANADIAN_VENUES)
    if not venue_ids:
        raise ValueError("Configure at least one Canadian-serving venue.")

    candles_by_asset = {}
    source_venue = None
    next_history_refresh = 0.0
    next_retrain = 0.0
    next_catalyst_refresh = 0.0
    next_whale_refresh = 0.0
    catalyst_service_available = False
    catalyst_error = "Not checked yet."
    market_events: list[dict[str, Any]] = []
    whale_flows: dict[str, dict[str, Any]] = {}
    while True:
        cycle_started = time.monotonic()
        snapshots, errors = fetch_canadian_spot_snapshots(
            venue_ids, assets=config.ARBITRAGE_ASSETS
        )
        for exchange_id, error in errors.items():
            logger.warning("Market data unavailable for %s: %s", exchange_id, error)
        now = time.time()
        if now >= next_catalyst_refresh:
            try:
                market_events = fetch_market_catalysts()
                engine.store_catalysts(market_events)
                catalyst_service_available = True
                high_impact = recent_high_impact_events(
                    market_events,
                    pause_minutes=config.CATALYST_PAUSE_MINUTES,
                )
                if high_impact:
                    logger.warning(
                        "%d recent high-impact headline(s); pausing new paper entries: %s",
                        len(high_impact),
                        high_impact[0]["title"],
                    )
            except Exception as exc:
                catalyst_service_available = False
                catalyst_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Catalyst feed unavailable; new paper entries will pause until it recovers: %s",
                    catalyst_error,
                )
            else:
                catalyst_error = ""
            next_catalyst_refresh = now + CATALYST_REFRESH_SECONDS

        if now >= next_whale_refresh:
            try:
                whale_flows, whale_prints, whale_errors = fetch_canadian_whale_flows(venue_ids)
                engine.store_whale_observations(whale_flows, whale_prints)
                for error in whale_errors:
                    logger.warning("Whale-flow data unavailable: %s", error)
            except Exception as exc:
                whale_flows = {}
                logger.warning(
                    "Large-trade flow scan failed; continuing without whale confirmation: %s",
                    exc,
                )
            next_whale_refresh = now + WHALE_REFRESH_SECONDS

        if now >= next_history_refresh:
            candles_by_asset = {}
            for exchange_id in venue_ids:
                try:
                    candles_by_asset, fee_rate = _download_history(exchange_id, candle_count)
                    if candles_by_asset:
                        source_venue = exchange_id
                        break
                except Exception as exc:
                    logger.warning("Could not load historical data from %s: %s", exchange_id, exc)
            next_history_refresh = now + 60 * 60
            if source_venue:
                logger.info("Using %d hourly CAD markets from %s.", len(candles_by_asset), source_venue)

        if snapshots and candles_by_asset and now >= next_retrain:
            try:
                historical_x, long_y, short_y, timestamps = prepare_historical_samples(
                    candles_by_asset, fee_rate
                )
                engine.train(historical_x, long_y, short_y, timestamps)
            except Exception:
                logger.exception("Historical model training failed.")
            gc.collect()
            next_retrain = now + MODEL_RETRAIN_SECONDS

        if snapshots and candles_by_asset and any(engine.models.values()):
            probabilities = engine.probabilities(candles_by_asset)
            recent_events = recent_high_impact_events(
                market_events,
                pause_minutes=config.CATALYST_PAUSE_MINUTES,
            )
            pause_entries = bool(recent_events) or not catalyst_service_available
            try:
                engine.update_paper_positions(
                    snapshots,
                    candles_by_asset,
                    probabilities,
                    whale_flows=whale_flows,
                    pause_new_entries=pause_entries,
                )
            except Exception:
                logger.exception("Paper trade cycle failed.")
            engine.update_service_metadata({
                "latest_probabilities": probabilities,
                "latest_prediction_at": datetime.now(timezone.utc).isoformat(),
            })

        engine.update_service_metadata({
            "last_cycle_at": datetime.now(timezone.utc).isoformat(),
            "source_venue": source_venue or "",
            "assets_loaded": sorted(candles_by_asset),
            "catalyst_available": catalyst_service_available,
            "catalyst_error": catalyst_error,
        })

        elapsed = time.monotonic() - cycle_started
        time.sleep(max(0, poll_seconds - elapsed))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Canadian crypto ML paper trader; no live orders are ever placed."
    )
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--candles", type=int, default=DEFAULT_CANDLE_COUNT)
    parser.add_argument("--database", default=None, help="SQLite paper journal path.")
    parser.add_argument("--model", default=None, help="Pickled local ML model path.")
    parser.add_argument("--capital-cad", type=float, default=10_000)
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=config.PAPER_MAX_RAM_GB,
        help="Optional hard cap for this process's estimated RAM budget; 0 uses automatic sizing.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="'cuda' requires one or two visible NVIDIA CUDA GPUs and CUDA-enabled PyTorch.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info(
        "Starting Canadian trading-pair ML paper trader. Exchange access is public/read-only; "
        "paper positions are local simulations."
    )
    run_forever(
        poll_seconds=args.poll_seconds,
        candle_count=args.candles,
        database_path=args.database,
        model_path=args.model,
        starting_capital_cad=args.capital_cad,
        device=args.device,
        max_ram_gb=args.max_ram_gb,
    )


if __name__ == "__main__":
    main()
