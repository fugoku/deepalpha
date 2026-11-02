"""Read-only, fee-aware arbitrage scans across Canadian-serving venues."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

import config

CANADIAN_VENUES = frozenset({"ndax", "kraken", "coinbase"})
MAX_ARBITRAGE_CAPITAL_CAD = 1000.0
FIAT_STABLE_QUOTES = frozenset({"CAD", "USD", "USDT", "USDC", "EUR", "GBP"})


def validate_fee_rate(fee_rate: float) -> float:
    """Reject fee settings that cannot represent a valid proportional fee."""
    fee_rate = float(fee_rate)
    if not 0 <= fee_rate < 1:
        raise ValueError("Taker fee must be at least 0 and less than 1.")
    return fee_rate


def validate_starting_capital(starting_cad: float) -> float:
    """Keep every displayed opportunity within the CAD 1,000 exposure cap."""
    amount = float(starting_cad)
    if not 0 < amount <= MAX_ARBITRAGE_CAPITAL_CAD:
        raise ValueError(
            f"Starting CAD capital must be greater than 0 and no more than "
            f"${MAX_ARBITRAGE_CAPITAL_CAD:,.0f}."
        )
    return amount


def validate_cost(amount: float, name: str) -> float:
    """Reject invalid cost inputs that could inflate estimated opportunity profit."""
    amount = float(amount)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"{name} must be a finite, non-negative CAD amount.")
    return amount


def _valid_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price > 0 else None


def _market_quotes(exchange_data: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Return valid quotes for active markets involving tracked coins or FX quotes."""
    tickers = exchange_data.get("tickers", {})
    markets = exchange_data.get("markets", {})
    quotes: dict[str, tuple[float, float]] = {}
    for symbol, ticker in tickers.items():
        market = markets.get(symbol, {})
        base, quote = market.get("base"), market.get("quote")
        if not market.get("spot", False) or market.get("active") is False:
            continue
        if not (
            base in config.COINS
            or quote in config.COINS
            or (base in FIAT_STABLE_QUOTES and quote in FIAT_STABLE_QUOTES)
        ):
            continue
        bid = _valid_price(ticker.get("bid"))
        ask = _valid_price(ticker.get("ask"))
        if bid is not None and ask is not None and bid <= ask:
            quotes[symbol] = (bid, ask)
    return quotes


def _conversion_rate(
    source: str,
    target: str,
    markets: dict[str, Any],
    quotes: dict[str, tuple[float, float]],
    fee_rate: float,
    *,
    max_hops: int = 4,
) -> tuple[float, int] | None:
    """Find a conservative executable conversion rate through available pairs."""
    if source == target:
        return 1.0, 0
    edges: dict[str, list[tuple[str, float]]] = {}
    fee_multiplier = 1 - fee_rate
    for symbol, (bid, ask) in quotes.items():
        market = markets.get(symbol, {})
        base, quote = market.get("base"), market.get("quote")
        if not base or not quote:
            continue
        edges.setdefault(base, []).append((quote, bid * fee_multiplier))
        edges.setdefault(quote, []).append((base, (1 / ask) * fee_multiplier))

    frontier = [(source, 1.0, frozenset({source}))]
    for hops in range(1, max_hops + 1):
        next_frontier = []
        found = []
        for currency, rate, visited in frontier:
            for next_currency, edge_rate in edges.get(currency, []):
                if next_currency in visited:
                    continue
                converted = rate * edge_rate
                if next_currency == target:
                    found.append((converted, hops))
                else:
                    next_frontier.append(
                        (next_currency, converted, visited | {next_currency})
                    )
        if found:
            return max(found, key=lambda route: route[0])
        frontier = next_frontier
    return None


def _book_levels(book: dict[str, Any], side: str) -> list[tuple[float, float]]:
    levels = []
    for level in book.get(side, []):
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        price, amount = _valid_price(level[0]), _valid_price(level[1])
        if price is not None and amount is not None:
            levels.append((price, amount))
    return sorted(levels, key=lambda level: level[0], reverse=side == "bids")


def _walk_cross_exchange_books(
    buy_book: dict[str, Any],
    sell_book: dict[str, Any],
    *,
    starting_cad: float,
    buy_fee: float,
    sell_fee: float,
) -> dict[str, Any] | None:
    """Estimate matched executable size while walking both venues' books."""
    asks = _book_levels(buy_book, "asks")
    bids = _book_levels(sell_book, "bids")
    if not asks or not bids:
        return None

    ask_index = bid_index = 0
    ask_remaining, bid_remaining = asks[0][1], bids[0][1]
    budget = starting_cad
    quantity = spent = proceeds = 0.0
    levels = []

    while ask_index < len(asks) and bid_index < len(bids) and budget > 0:
        ask, bid = asks[ask_index][0], bids[bid_index][0]
        unit_cost = ask * (1 + buy_fee)
        unit_proceeds = bid * (1 - sell_fee)
        if unit_proceeds <= unit_cost:
            break
        filled = min(ask_remaining, bid_remaining, budget / unit_cost)
        if filled <= 0:
            break
        level_spend = filled * unit_cost
        level_proceeds = filled * unit_proceeds
        quantity += filled
        spent += level_spend
        proceeds += level_proceeds
        budget -= level_spend
        levels.append({
            "buy_ask_cad": ask,
            "sell_bid_cad": bid,
            "base_quantity": filled,
            "net_before_fixed_costs_cad": level_proceeds - level_spend,
        })
        ask_remaining -= filled
        bid_remaining -= filled
        if ask_remaining <= 1e-12:
            ask_index += 1
            if ask_index < len(asks):
                ask_remaining = asks[ask_index][1]
        if bid_remaining <= 1e-12:
            bid_index += 1
            if bid_index < len(bids):
                bid_remaining = bids[bid_index][1]

    if quantity <= 0:
        return None
    return {
        "base_quantity": quantity,
        "buy_spent_cad": spent,
        "sell_proceeds_cad": proceeds,
        "unspent_cad": max(0.0, starting_cad - spent),
        "depth_levels": levels,
    }


def find_triangular_opportunities(
    exchange_id: str,
    exchange_data: dict[str, Any],
    *,
    starting_cad: float = config.ARBITRAGE_CAPITAL_CAD,
    fixed_fee_cad_per_order: float = config.ARBITRAGE_FIXED_FEE_CAD_PER_ORDER,
) -> list[dict[str, Any]]:
    """Find three-market CAD round trips, including taker fees and configured costs."""
    if exchange_id not in CANADIAN_VENUES:
        raise ValueError("Exchange is not in the Canadian arbitrage allowlist.")
    starting_cad = validate_starting_capital(starting_cad)
    fixed_fee_cad_per_order = validate_cost(
        fixed_fee_cad_per_order, "Fixed fee per order"
    )
    fee = validate_fee_rate(exchange_data["fee_rate"])
    quotes = _market_quotes(exchange_data)
    markets = exchange_data.get("markets", {})
    edges: dict[str, list[tuple[str, float, str, str]]] = {}

    for symbol, (bid, ask) in quotes.items():
        market = markets[symbol]
        base, quote = market["base"], market["quote"]
        if base not in config.COINS and quote not in config.COINS:
            continue
        edges.setdefault(base, []).append((quote, bid * (1 - fee), symbol, "sell"))
        edges.setdefault(quote, []).append((base, (1 / ask) * (1 - fee), symbol, "buy"))

    opportunities = []
    for first, rate1, symbol1, side1 in edges.get("CAD", []):
        if first == "CAD":
            continue
        for second, rate2, symbol2, side2 in edges.get(first, []):
            if second in {"CAD", first}:
                continue
            for third, rate3, symbol3, side3 in edges.get(second, []):
                if third != "CAD" or len({symbol1, symbol2, symbol3}) != 3:
                    continue
                ending_cad = starting_cad * rate1 * rate2 * rate3
                net_profit = (
                    ending_cad - starting_cad - 3 * fixed_fee_cad_per_order
                )
                if net_profit <= 0:
                    continue
                opportunities.append({
                    "type": "Triangular",
                    "exchange": exchange_id,
                    "route": f"CAD -> {first} -> {second} -> CAD",
                    "legs": (
                        f"{side1} {symbol1}",
                        f"{side2} {symbol2}",
                        f"{side3} {symbol3}",
                    ),
                    "starting_cad": starting_cad,
                    "ending_cad": ending_cad,
                    "net_profit_cad": net_profit,
                    "net_return_pct": net_profit / starting_cad * 100 if starting_cad else 0,
                    "taker_fee_pct_per_leg": fee * 100,
                })
    return sorted(opportunities, key=lambda item: item["net_profit_cad"], reverse=True)


def find_cross_exchange_opportunities(
    exchange_data: dict[str, dict[str, Any]],
    *,
    starting_cad: float = config.ARBITRAGE_CAPITAL_CAD,
    fixed_fee_cad_per_order: float = config.ARBITRAGE_FIXED_FEE_CAD_PER_ORDER,
    rebalance_cost_cad: float = config.ARBITRAGE_REBALANCE_COST_CAD,
) -> list[dict[str, Any]]:
    """Compare same-asset CAD spot bids and asks on separate Canadian-serving venues."""
    starting_cad = validate_starting_capital(starting_cad)
    fixed_fee_cad_per_order = validate_cost(
        fixed_fee_cad_per_order, "Fixed fee per order"
    )
    rebalance_cost_cad = validate_cost(rebalance_cost_cad, "Rebalance cost")
    by_venue = {
        exchange_id: _market_quotes(data)
        for exchange_id, data in exchange_data.items()
        if exchange_id in CANADIAN_VENUES
    }

    opportunities = []
    venue_ids = sorted(by_venue)
    for i, first_exchange in enumerate(venue_ids):
        first_data = exchange_data[first_exchange]
        first_fee = validate_fee_rate(first_data["fee_rate"])
        for second_exchange in venue_ids[i + 1:]:
            second_data = exchange_data[second_exchange]
            second_fee = validate_fee_rate(second_data["fee_rate"])

            for symbol in by_venue[first_exchange].keys() & by_venue[second_exchange].keys():
                market = first_data.get("markets", {}).get(symbol, {})
                if market.get("base") not in config.COINS:
                    continue
                first_bid, first_ask = by_venue[first_exchange][symbol]
                second_bid, second_ask = by_venue[second_exchange][symbol]
                for buy_exchange, sell_exchange, buy_ask, sell_bid, buy_fee, sell_fee in (
                    (first_exchange, second_exchange, first_ask, second_bid, first_fee, second_fee),
                    (second_exchange, first_exchange, second_ask, first_bid, second_fee, first_fee),
                ):
                    if buy_ask >= sell_bid:
                        continue
                    buy_data = exchange_data[buy_exchange]
                    sell_data = exchange_data[sell_exchange]
                    buy_quotes = by_venue[buy_exchange]
                    sell_quotes = by_venue[sell_exchange]
                    buy_conversion_route = _conversion_rate(
                        "CAD",
                        market["quote"],
                        buy_data.get("markets", {}),
                        buy_quotes,
                        buy_fee,
                    )
                    sell_conversion_route = _conversion_rate(
                        market["quote"],
                        "CAD",
                        sell_data.get("markets", {}),
                        sell_quotes,
                        sell_fee,
                    )
                    if (
                        not buy_conversion_route
                        or not sell_conversion_route
                    ):
                        continue
                    buy_conversion, buy_conversion_hops = buy_conversion_route
                    sell_conversion, sell_conversion_hops = sell_conversion_route
                    starting_quote = starting_cad * buy_conversion
                    depth_result = None
                    buy_book = buy_data.get("order_books", {}).get(symbol)
                    sell_book = sell_data.get("order_books", {}).get(symbol)
                    depth_available = bool(buy_book and sell_book)
                    if depth_available:
                        buy_quote_to_cad_route = _conversion_rate(
                            market["quote"],
                            "CAD",
                            buy_data.get("markets", {}),
                            buy_quotes,
                            buy_fee,
                        )
                        if not buy_quote_to_cad_route:
                            continue
                        buy_quote_to_cad, buy_quote_to_cad_hops = (
                            buy_quote_to_cad_route
                        )
                        depth_result = _walk_cross_exchange_books(
                            buy_book,
                            sell_book,
                            starting_cad=starting_quote,
                            buy_fee=buy_fee,
                            sell_fee=sell_fee,
                        )
                        if depth_result is None:
                            continue
                    if depth_result:
                        buy_spent = depth_result["buy_spent_cad"]
                        proceeds = depth_result["sell_proceeds_cad"]
                        leftover_quote = depth_result["unspent_cad"]
                        leftover_cad = leftover_quote * buy_quote_to_cad
                        conversion_orders = (
                            buy_conversion_hops
                            + sell_conversion_hops
                            + (buy_quote_to_cad_hops if leftover_quote > 0 else 0)
                        )
                        net_profit = (
                            proceeds * sell_conversion
                            + leftover_cad - starting_cad
                            - (2 + conversion_orders) * fixed_fee_cad_per_order
                            - rebalance_cost_cad
                        )
                        ending_cad = starting_cad + net_profit
                    else:
                        base_bought = starting_quote / buy_ask * (1 - buy_fee)
                        proceeds = base_bought * sell_bid * (1 - sell_fee)
                        buy_spent = starting_quote
                        net_profit = (
                            proceeds * sell_conversion - starting_cad
                            - (
                                2 + buy_conversion_hops + sell_conversion_hops
                            ) * fixed_fee_cad_per_order
                            - rebalance_cost_cad
                        )
                        ending_cad = starting_cad + net_profit
                    if net_profit <= 0:
                        continue
                    opportunities.append({
                        "type": "Cross-exchange",
                        "symbol": symbol,
                        "buy_exchange": buy_exchange,
                        "sell_exchange": sell_exchange,
                        "quote_currency": market["quote"],
                        "buy_ask": buy_ask,
                        "sell_bid": sell_bid,
                        "starting_cad": starting_cad,
                        "ending_cad": ending_cad,
                        "net_profit_cad": net_profit,
                        "net_return_pct": net_profit / starting_cad * 100 if starting_cad else 0,
                        "buy_taker_fee_pct": buy_fee * 100,
                        "sell_taker_fee_pct": sell_fee * 100,
                        "execution_basis": "order-book depth" if depth_available else "top-of-book estimate",
                        "buy_spent_quote": buy_spent,
                        "buy_spent_cad": buy_spent / buy_conversion,
                        "starting_quote": starting_quote,
                        "quote_to_cad_rate": sell_conversion,
                        "base_quantity": (
                            depth_result["base_quantity"] if depth_result else base_bought
                        ),
                        "depth_levels": depth_result["depth_levels"] if depth_result else [],
                    })
    return sorted(opportunities, key=lambda item: item["net_profit_cad"], reverse=True)


def fetch_canadian_spot_snapshots(
    exchange_ids: tuple[str, ...] = config.CANADIAN_ARBITRAGE_EXCHANGES,
    *,
    include_order_books: bool = False,
    assets: frozenset[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Fetch public CAD spot tickers from configured Canada-serving venues."""
    import ccxt

    snapshots = {}
    errors = {}
    for exchange_id in exchange_ids:
        if exchange_id not in CANADIAN_VENUES:
            errors[exchange_id] = "Exchange is not in the Canadian arbitrage allowlist."
            continue
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            errors[exchange_id] = "Exchange adapter is not available in the installed CCXT version."
            continue
        try:
            tracked_assets = config.COINS if assets is None else assets
            client = exchange_class({"enableRateLimit": True, "timeout": 20000})
            markets = client.load_markets()
            symbols = [
                symbol
                for symbol, market in markets.items()
                if market.get("spot")
                and market.get("active") is not False
                and (
                    market.get("base") in tracked_assets
                    or market.get("quote") in tracked_assets
                    or (
                        market.get("base") in FIAT_STABLE_QUOTES
                        and market.get("quote") in FIAT_STABLE_QUOTES
                    )
                )
            ]
            if assets is not None:
                symbols = [
                    symbol for symbol in symbols
                    if markets[symbol].get("base") in tracked_assets
                    and markets[symbol].get("quote") == "CAD"
                ]
            if not symbols:
                errors[exchange_id] = "No configured Canadian crypto trading pairs are available."
                continue
            tickers = client.fetch_tickers(symbols)
            snapshots[exchange_id] = {
                "markets": markets,
                "tickers": tickers,
                "order_books": {},
                "fee_rate": config.ARBITRAGE_TAKER_FEES.get(exchange_id, 0.005),
                "as_of": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            errors[exchange_id] = f"{type(exc).__name__}: {exc}"
    if include_order_books:
        order_book_clients = {}
        for index, buy_exchange in enumerate(sorted(snapshots)):
            buy_data = snapshots[buy_exchange]
            buy_quotes = _market_quotes(buy_data)
            for sell_exchange in sorted(snapshots)[index + 1:]:
                sell_data = snapshots[sell_exchange]
                sell_quotes = _market_quotes(sell_data)
                for symbol in buy_quotes.keys() & sell_quotes.keys():
                    market = buy_data["markets"].get(symbol, {})
                    if market.get("base") not in config.COINS:
                        continue
                    buy_fee = validate_fee_rate(buy_data["fee_rate"])
                    sell_fee = validate_fee_rate(sell_data["fee_rate"])
                    buy_bid, buy_ask = buy_quotes[symbol]
                    sell_bid, sell_ask = sell_quotes[symbol]
                    profitable_spread = (
                        buy_ask * (1 + buy_fee) < sell_bid * (1 - sell_fee)
                        or sell_ask * (1 + sell_fee) < buy_bid * (1 - buy_fee)
                    )
                    if not profitable_spread:
                        continue
                    for exchange_id in (buy_exchange, sell_exchange):
                        if symbol in snapshots[exchange_id]["order_books"]:
                            continue
                        try:
                            if exchange_id not in order_book_clients:
                                order_book_clients[exchange_id] = getattr(ccxt, exchange_id)(
                                    {"enableRateLimit": True, "timeout": 20000}
                                )
                                order_book_clients[exchange_id].load_markets()
                            client = order_book_clients[exchange_id]
                            snapshots[exchange_id]["order_books"][symbol] = (
                                client.fetch_order_book(symbol, limit=20)
                            )
                        except Exception as exc:
                            errors[f"{exchange_id} {symbol} order book"] = (
                                f"{type(exc).__name__}: {exc}"
                            )
    return snapshots, errors


def fetch_canadian_price_history(
    exchange_id: str,
    asset: str,
    timeframe: str = "1h",
    limit: int = 168,
) -> list[list[float]]:
    """Fetch OHLCV chart candles for one approved crypto/CAD spot market."""
    import ccxt

    if exchange_id not in CANADIAN_VENUES:
        raise ValueError("Exchange is not in the Canadian arbitrage allowlist.")
    if asset not in config.COINS:
        raise ValueError("Asset is not in the configured crypto chart universe.")
    if timeframe not in {"1h", "4h", "1d"}:
        raise ValueError("Chart timeframe must be 1h, 4h, or 1d.")
    client = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 20000})
    markets = client.load_markets()
    symbol = f"{asset}/CAD"
    market = markets.get(symbol)
    if not market or not market.get("spot") or market.get("active") is False:
        raise ValueError(f"{symbol} is not an active CAD spot market on {exchange_id}.")
    candles = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not candles:
        raise ValueError(f"No {timeframe} candles returned for {symbol} on {exchange_id}.")
    return candles
