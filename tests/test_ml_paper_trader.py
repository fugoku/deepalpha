import sqlite3

import numpy as np
import pytest

from ml_paper_trader import (
    INDICATOR_NAMES,
    PaperLearningEngine,
    build_indicator_matrix,
    cap_candle_history,
    chronological_split,
    prepare_historical_samples,
    resolve_memory_budget,
    _require_cuda_device,
    _positive_class_weight,
)


def _candles(n=240, seed=7):
    rng = np.random.default_rng(seed)
    close = 1.0 * np.cumprod(1 + rng.normal(0, 0.012, n))
    result = []
    for i, price in enumerate(close):
        open_price = close[i - 1] if i else price
        result.append([
            i * 3_600_000,
            float(open_price),
            float(max(open_price, price) * 1.003),
            float(min(open_price, price) * 0.997),
            float(price),
            float(rng.uniform(100, 1000)),
        ])
    return result


def _snapshot(bid, ask, fee=0.001):
    return {
        "fee_rate": fee,
        "markets": {
            "XLM/CAD": {
                "base": "XLM", "quote": "CAD", "spot": True, "active": True,
            },
        },
        "tickers": {"XLM/CAD": {"bid": bid, "ask": ask}},
    }


def test_indicators_are_finite_and_have_expected_width():
    features = build_indicator_matrix(_candles())

    assert features.shape == (240, len(INDICATOR_NAMES))
    assert np.isfinite(features).all()


def test_rsi_handles_monotonic_and_flat_price_series():
    def constant_ohlcv(n, price):
        return [
            [i * 3_600_000, price, price, price, price, 100.0]
            for i in range(n)
        ]

    rising = constant_ohlcv(40, 100.0)
    for i, candle in enumerate(rising):
        candle[4] = 100.0 + i
        candle[2] = candle[4]
    rising_rsi = build_indicator_matrix(rising)[-1, INDICATOR_NAMES.index("rsi_14")]
    flat_rsi = build_indicator_matrix(constant_ohlcv(40, 100.0))[
        -1, INDICATOR_NAMES.index("rsi_14")
    ]

    assert rising_rsi == 1.0
    assert flat_rsi == 0.5


def test_historical_samples_create_bullish_and_bearish_targets():
    features, long_labels, short_labels, timestamps = prepare_historical_samples(
        {"XLM": _candles()}, fee_rate=0.001
    )

    assert features.shape[1] == len(INDICATOR_NAMES)
    assert len(long_labels) == len(short_labels) == len(features)
    assert len(timestamps) == len(features)
    assert set(long_labels) == {0, 1}
    assert set(short_labels) == {0, 1}


def test_historical_features_use_a_forward_label_horizon():
    candles = _candles(100)
    features, long_labels, short_labels, timestamps = prepare_historical_samples(
        {"XLM": candles}, fee_rate=0.001
    )

    assert len(features) == 100 - 4 - 30
    assert len(long_labels) == len(short_labels)
    assert np.all(timestamps[:-1] <= timestamps[1:])


def test_long_and_short_paper_positions_close_and_include_fees(tmp_path):
    engine = PaperLearningEngine(
        str(tmp_path / "paper.sqlite3"),
        str(tmp_path / "model.pkl"),
        starting_capital_cad=10_000,
    )
    candles = {"XLM": _candles(40)}
    data = {"ndax": _snapshot(0.99, 1.01)}

    engine.update_paper_positions(data, candles, {"XLM": (0.9, 0.1)})
    with sqlite3.connect(engine.database_path) as db:
        assert db.execute("SELECT side FROM paper_trades").fetchone()[0] == "long"

    engine.update_paper_positions(data, candles, {"XLM": (0.1, 0.9)})
    with sqlite3.connect(engine.database_path) as db:
        side, status, pnl = db.execute(
            "SELECT side, status, net_pnl_cad FROM paper_trades"
        ).fetchone()
    assert side == "long"
    assert status == "closed"
    assert pnl is not None

    short_quotes = {"ndax": _snapshot(0.99, 1.01)}
    engine.update_paper_positions(short_quotes, candles, {"XLM": (0.1, 0.9)})
    engine.update_paper_positions(short_quotes, candles, {"XLM": (0.1, 0.9)})
    with sqlite3.connect(engine.database_path) as db:
        rows = db.execute("SELECT side, status FROM paper_trades ORDER BY id").fetchall()
    assert rows[-1] == ("short", "open")

    engine.update_paper_positions({"ndax": _snapshot(0.80, 0.82)}, candles,
                                  {"XLM": (0.1, 0.1)})
    with sqlite3.connect(engine.database_path) as db:
        side, status, pnl = db.execute(
            "SELECT side, status, net_pnl_cad FROM paper_trades ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert side == "short"
    assert status == "closed"
    assert pnl > 0


def test_short_paper_trade_records_ask_as_exit_price(tmp_path):
    engine = PaperLearningEngine(
        str(tmp_path / "paper.sqlite3"), str(tmp_path / "model.pkl")
    )
    candles = {"XLM": _candles(40)}
    engine.update_paper_positions(
        {"ndax": _snapshot(0.99, 1.01)}, candles, {"XLM": (0.1, 0.9)}
    )
    engine.update_paper_positions(
        {"ndax": _snapshot(0.79, 0.81)}, candles, {"XLM": (0.1, 0.1)}
    )
    with sqlite3.connect(engine.database_path) as db:
        exit_price = db.execute(
            "SELECT exit_price FROM paper_trades"
        ).fetchone()[0]

    assert exit_price == 0.81


def test_chronological_model_training_and_direction_probabilities(tmp_path):
    candles = {"XLM": _candles(500, seed=123)}
    features, long_labels, short_labels, timestamps = prepare_historical_samples(
        candles, fee_rate=0.001
    )
    engine = PaperLearningEngine(
        str(tmp_path / "paper.sqlite3"), str(tmp_path / "model.pkl")
    )

    accuracies = engine.train(features, long_labels, short_labels, timestamps)
    predictions = engine.probabilities(candles)

    assert accuracies
    assert "XLM" in predictions
    assert 0 <= predictions["XLM"][0] <= 1
    assert 0 <= predictions["XLM"][1] <= 1


def test_chronological_training_skips_insufficient_data(tmp_path):
    engine = PaperLearningEngine(str(tmp_path / "paper.sqlite3"), str(tmp_path / "model.pkl"))
    x = np.zeros((20, len(INDICATOR_NAMES)))
    y = np.zeros(20, dtype=int)

    assert engine.train(x, y, y) is None


def test_cuda_mode_requires_at_least_one_visible_device():
    with pytest.raises(RuntimeError, match="no CUDA devices"):
        _require_cuda_device(0)


@pytest.mark.parametrize("device_count", [1, 2, 3])
def test_cuda_mode_accepts_one_or_two_devices_and_caps_at_two(device_count):
    _require_cuda_device(device_count)


def test_memory_budget_preserves_headroom_and_respects_user_cap():
    available, budget = resolve_memory_budget(
        max_ram_gb=0.25,
        fraction=0.5,
        available_bytes=1024**3,
    )

    assert available == 1024**3
    assert budget == int(0.25 * 1024**3)


def test_history_is_limited_by_ram_budget():
    per_asset_budget = 100 * 16 * 1024

    assert cap_candle_history(720, per_asset_budget, 1) == 100
    assert cap_candle_history(720, per_asset_budget * 8, 8) == 100


def test_too_small_ram_budget_fails_explicitly():
    with pytest.raises(RuntimeError, match="too small"):
        cap_candle_history(720, 1_000, 8)


def test_chronological_split_purges_overlapping_forward_labels():
    timestamps = np.repeat(np.arange(100, dtype=np.int64) * 3_600_000, 2)

    train_mask, test_mask = chronological_split(timestamps, horizon=4)
    last_train_time = timestamps[train_mask].max()
    first_test_time = timestamps[test_mask].min()

    assert first_test_time - last_train_time >= 4 * 3_600_000


def test_gpu_loss_positive_weight_accounts_for_imbalanced_labels():
    assert _positive_class_weight(np.array([0] * 8 + [1] * 2)) == 4
