"""Exchange-observed large-trade flow estimates; not wallet or identity tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import config
from canadian_arbitrage import CANADIAN_VENUES


def summarize_whale_trades(
    trades: list[dict[str, Any]],
    *,
    exchange_id: str,
    symbol: str,
    minimum_trade_cad: float = config.WHALE_MIN_TRADE_CAD,
    median_multiple: float = config.WHALE_MEDIAN_MULTIPLE,
    now_ms: int | None = None,
    max_age_ms: int = 60 * 60 * 1000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Detect unusually large, recent CAD-quoted prints and signed taker flow."""
    if minimum_trade_cad <= 0 or median_multiple <= 1:
        raise ValueError("Whale thresholds must be positive and median multiple must exceed 1.")
    current_ms = now_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    parsed = []
    for trade in trades:
        try:
            price = float(trade.get("price") or 0)
            amount = float(trade.get("amount") or 0)
            timestamp = int(trade.get("timestamp") or current_ms)
        except (TypeError, ValueError):
            continue
        side = str(trade.get("side") or "").lower()
        if price <= 0 or amount <= 0 or side not in {"buy", "sell"}:
            continue
        if timestamp > current_ms + 60_000 or current_ms - timestamp > max_age_ms:
            continue
        parsed.append({
            "timestamp_ms": timestamp,
            "side": side,
            "notional_cad": price * amount,
        })

    if not parsed:
        return {
            "exchange": exchange_id,
            "symbol": symbol,
            "threshold_cad": minimum_trade_cad,
            "buy_notional_cad": 0.0,
            "sell_notional_cad": 0.0,
            "net_flow_cad": 0.0,
            "flow_ratio": 0.0,
            "whale_count": 0,
        }, []

    median_notional = sorted(t["notional_cad"] for t in parsed)[len(parsed) // 2]
    threshold = max(minimum_trade_cad, median_notional * median_multiple)
    whales = [trade for trade in parsed if trade["notional_cad"] >= threshold]
    buy_value = sum(t["notional_cad"] for t in whales if t["side"] == "buy")
    sell_value = sum(t["notional_cad"] for t in whales if t["side"] == "sell")
    gross = buy_value + sell_value
    summary = {
        "exchange": exchange_id,
        "symbol": symbol,
        "threshold_cad": threshold,
        "buy_notional_cad": buy_value,
        "sell_notional_cad": sell_value,
        "net_flow_cad": buy_value - sell_value,
        "flow_ratio": (buy_value - sell_value) / gross if gross else 0.0,
        "whale_count": len(whales),
    }
    prints = [
        {
            **trade,
            "exchange": exchange_id,
            "symbol": symbol,
            "threshold_cad": threshold,
            "flow_destination": (
                f"{exchange_id} {trade['side']} aggressor: "
                f"{'CAD into crypto' if trade['side'] == 'buy' else 'crypto into CAD'}"
            ),
        }
        for trade in whales
    ]
    return summary, prints


def aggregate_whale_flows(
    summaries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate venue-level large-trade imbalance by asset."""
    aggregate: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        asset = summary["symbol"].split("/")[0]
        item = aggregate.setdefault(asset, {
            "buy_notional_cad": 0.0,
            "sell_notional_cad": 0.0,
            "net_flow_cad": 0.0,
            "flow_ratio": 0.0,
            "whale_count": 0,
            "venues": [],
        })
        item["buy_notional_cad"] += summary["buy_notional_cad"]
        item["sell_notional_cad"] += summary["sell_notional_cad"]
        item["net_flow_cad"] += summary["net_flow_cad"]
        item["whale_count"] += summary["whale_count"]
        item["venues"].append(summary["exchange"])
    for item in aggregate.values():
        gross = item["buy_notional_cad"] + item["sell_notional_cad"]
        item["flow_ratio"] = item["net_flow_cad"] / gross if gross else 0.0
        item["venues"] = sorted(set(item["venues"]))
    return aggregate


def fetch_canadian_whale_flows(
    exchange_ids: tuple[str, ...] = config.CANADIAN_ARBITRAGE_EXCHANGES,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Fetch recent public trades on active configured CAD crypto spot markets."""
    import ccxt

    summaries = []
    prints = []
    errors = []
    for exchange_id in exchange_ids:
        if exchange_id not in CANADIAN_VENUES:
            errors.append(f"{exchange_id}: not in the Canadian venue allowlist")
            continue
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            errors.append(f"{exchange_id}: CCXT adapter unavailable")
            continue
        try:
            client = exchange_class({"enableRateLimit": True, "timeout": 20000})
            markets = client.load_markets()
        except Exception as exc:
            errors.append(f"{exchange_id}: {type(exc).__name__}: {exc}")
            continue
        for asset in sorted(config.ARBITRAGE_ASSETS):
            symbol = f"{asset}/CAD"
            market = markets.get(symbol)
            if not market or not market.get("spot") or market.get("active") is False:
                continue
            try:
                trades = client.fetch_trades(symbol, limit=100)
                summary, large_prints = summarize_whale_trades(
                    trades,
                    exchange_id=exchange_id,
                    symbol=symbol,
                )
                summaries.append(summary)
                prints.extend(large_prints)
            except Exception as exc:
                errors.append(f"{exchange_id} {symbol}: {type(exc).__name__}: {exc}")
    return aggregate_whale_flows(summaries), prints, errors
