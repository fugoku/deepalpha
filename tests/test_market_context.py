from datetime import datetime, timedelta, timezone

from catalyst_monitor import classify_headline, recent_high_impact_events
from ml_paper_trader import PaperLearningEngine, whale_direction_confirmation
from whale_flow import summarize_whale_trades


def test_leader_speech_about_crypto_is_classified_as_high_impact():
    event = classify_headline("President speaks about cryptocurrency policy")

    assert event is not None
    assert event["category"] == "leader_speech"
    assert event["impact"] == "high"
    assert event["mentions_speech"] is True


def test_recent_high_impact_filter_respects_pause_window():
    now = datetime.now(timezone.utc)
    events = [
        {"impact": "high", "published_at": (now - timedelta(minutes=5)).isoformat()},
        {"impact": "high", "published_at": (now - timedelta(minutes=90)).isoformat()},
        {"impact": "watch", "published_at": (now - timedelta(minutes=2)).isoformat()},
    ]

    assert len(recent_high_impact_events(events, now=now, pause_minutes=60)) == 1


def test_whale_flow_counts_recent_large_prints_and_records_observed_direction():
    now_ms = 1_000_000
    trades = [
        {"timestamp": now_ms - 1_000, "side": "buy", "price": 1, "amount": 100},
        {"timestamp": now_ms - 2_000, "side": "sell", "price": 1, "amount": 100},
        {"timestamp": now_ms - 2_500, "side": "buy", "price": 1, "amount": 100},
        {"timestamp": now_ms - 2_700, "side": "sell", "price": 1, "amount": 100},
        {"timestamp": now_ms - 3_000, "side": "buy", "price": 1, "amount": 2_000},
        {"timestamp": now_ms - 4_000, "side": "buy", "price": 1, "amount": 3_000},
        {"timestamp": now_ms - 4_000_000, "side": "buy", "price": 1, "amount": 50_000},
    ]

    summary, prints = summarize_whale_trades(
        trades,
        exchange_id="ndax",
        symbol="XLM/CAD",
        minimum_trade_cad=500,
        median_multiple=5,
        now_ms=now_ms,
    )

    assert summary["whale_count"] == 2
    assert summary["net_flow_cad"] == 5_000
    assert len(prints) == 2
    assert all("ndax buy aggressor" in trade["flow_destination"] for trade in prints)


def test_whale_flow_gates_only_strong_opposing_imbalance():
    assert whale_direction_confirmation(
        0.8, 0.2, {"whale_count": 3, "flow_ratio": -0.8}
    )[0] is None
    assert whale_direction_confirmation(
        0.8, 0.2, {"whale_count": 3, "flow_ratio": 0.8}
    )[0] == "long"


def test_catalyst_and_whale_observations_are_persisted(tmp_path):
    engine = PaperLearningEngine(
        str(tmp_path / "paper.sqlite3"),
        str(tmp_path / "model.pkl"),
    )
    event = {
        "url": "https://example.test/event",
        "title": "Crypto market event",
        "source": "example.test",
        "category": "market_news",
        "impact": "watch",
        "tone": 0.0,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    flow = {
        "buy_notional_cad": 10_000.0,
        "sell_notional_cad": 2_000.0,
        "net_flow_cad": 8_000.0,
        "flow_ratio": 0.667,
        "whale_count": 2,
        "venues": ["ndax"],
    }
    trade = {
        "exchange": "ndax",
        "symbol": "XLM/CAD",
        "timestamp_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
        "side": "buy",
        "notional_cad": 5_000.0,
        "threshold_cad": 1_000.0,
        "flow_destination": "ndax buy aggressor: CAD into crypto",
    }

    engine.store_catalysts([event])
    engine.store_whale_observations({"XLM": flow}, [trade])

    assert engine.recent_catalysts()[0]["url"] == event["url"]
    flows, prints = engine.latest_whale_observations()
    assert flows[0]["asset"] == "XLM"
    assert prints[0]["flow_destination"] == trade["flow_destination"]
