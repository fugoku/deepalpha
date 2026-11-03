import sys
from types import SimpleNamespace

import pytest

from canadian_arbitrage import (
    fetch_canadian_spot_snapshots,
    find_cross_exchange_opportunities,
    find_triangular_opportunities,
    validate_cost,
    validate_fee_rate,
    validate_starting_capital,
)


def _venue(fee, tickers):
    markets = {}
    market_tickers = {}
    for symbol, (base, quote, bid, ask) in tickers.items():
        markets[symbol] = {"base": base, "quote": quote, "spot": True, "active": True}
        market_tickers[symbol] = {"bid": bid, "ask": ask}
    return {"fee_rate": fee, "markets": markets, "tickers": market_tickers}


def test_cross_exchange_profit_subtracts_both_taker_fees_and_rebalance_cost():
    data = {
        "ndax": _venue(0.001, {"XLM/CAD": ("XLM", "CAD", 0.99, 1.00)}),
        "kraken": _venue(0.001, {"XLM/CAD": ("XLM", "CAD", 1.02, 1.03)}),
    }

    opportunities = find_cross_exchange_opportunities(
        data,
        starting_cad=1000,
        rebalance_cost_cad=2,
    )

    assert len(opportunities) == 1
    expected = (1000 / 1.00 * 0.999 * 1.02 * 0.999) - 1000 - 2
    assert opportunities[0]["buy_exchange"] == "ndax"
    assert opportunities[0]["sell_exchange"] == "kraken"
    assert opportunities[0]["net_profit_cad"] == pytest.approx(expected)


def test_cross_exchange_scans_non_cad_pair_and_converts_profit_to_cad():
    first = _venue(0, {
        "XLM/USD": ("XLM", "USD", 0.99, 1.00),
        "USD/CAD": ("USD", "CAD", 1.35, 1.36),
    })
    second = _venue(0, {
        "XLM/USD": ("XLM", "USD", 1.02, 1.03),
        "USD/CAD": ("USD", "CAD", 1.34, 1.35),
    })

    opportunities = find_cross_exchange_opportunities(
        {"ndax": first, "kraken": second},
        starting_cad=1000,
        fixed_fee_cad_per_order=0,
        rebalance_cost_cad=0,
    )

    result = next(item for item in opportunities if item["symbol"] == "XLM/USD")
    assert result["quote_currency"] == "USD"
    assert result["buy_exchange"] == "ndax"
    assert result["sell_exchange"] == "kraken"
    assert result["net_profit_cad"] == pytest.approx(
        (1000 / 1.36 * 1.02 * 1.34) - 1000
    )
    assert result["starting_cad"] == 1000


def test_non_cad_depth_walk_converts_unspent_quote_back_to_cad():
    buy = _venue(0, {
        "XLM/USD": ("XLM", "USD", 0.99, 1.00),
        "USD/CAD": ("USD", "CAD", 1.35, 1.35),
    })
    sell = _venue(0, {
        "XLM/USD": ("XLM", "USD", 1.04, 1.05),
        "USD/CAD": ("USD", "CAD", 1.35, 1.35),
    })
    buy["order_books"] = {
        "XLM/USD": {"asks": [[1.00, 10]], "bids": []}
    }
    sell["order_books"] = {
        "XLM/USD": {"asks": [], "bids": [[1.05, 10]]}
    }

    result = find_cross_exchange_opportunities(
        {"ndax": buy, "kraken": sell},
        starting_cad=1000,
        fixed_fee_cad_per_order=0,
        rebalance_cost_cad=0,
    )[0]

    assert result["net_profit_cad"] == pytest.approx(0.675)


def test_scanner_keeps_small_positive_profit_without_minimum_cutoff():
    data = {
        "ndax": _venue(0, {"XLM/CAD": ("XLM", "CAD", 1.00001, 1.00002)}),
        "kraken": _venue(0, {"XLM/CAD": ("XLM", "CAD", 1.00003, 1.00004)}),
    }

    opportunities = find_cross_exchange_opportunities(
        data,
        starting_cad=1000,
        fixed_fee_cad_per_order=0,
        rebalance_cost_cad=0,
    )

    assert len(opportunities) == 1
    assert 0 < opportunities[0]["net_profit_cad"] < 0.01


@pytest.mark.parametrize("capital", [1000.01, 5000])
def test_scanner_rejects_capital_above_one_thousand_cad(capital):
    with pytest.raises(ValueError, match="no more than \\$1,000"):
        validate_starting_capital(capital)


@pytest.mark.parametrize("capital", [0, -1])
def test_scanner_rejects_nonpositive_capital(capital):
    with pytest.raises(ValueError, match="greater than 0"):
        validate_starting_capital(capital)


def test_scanner_allows_capital_up_to_one_thousand_cad():
    assert validate_starting_capital(1000) == 1000


@pytest.mark.parametrize("cost", [-0.01, float("inf"), float("nan")])
def test_scanner_rejects_invalid_configured_costs(cost):
    with pytest.raises(ValueError, match="finite, non-negative CAD amount"):
        validate_cost(cost, "Fixed fee per order")


def test_cross_exchange_rejects_non_canadian_venue():
    data = {
        "binance": _venue(0, {"XLM/CAD": ("XLM", "CAD", 1.1, 1.11)}),
        "ndax": _venue(0, {"XLM/CAD": ("XLM", "CAD", 1.0, 1.01)}),
    }

    assert find_cross_exchange_opportunities(data, rebalance_cost_cad=0) == []


def test_cross_exchange_includes_all_configured_coin_pairs():
    data = {
        "ndax": _venue(0, {"BTC/CAD": ("BTC", "CAD", 100, 101)}),
        "kraken": _venue(0, {"BTC/CAD": ("BTC", "CAD", 110, 111)}),
    }

    opportunities = find_cross_exchange_opportunities(data, rebalance_cost_cad=0)

    assert len(opportunities) == 1
    assert opportunities[0]["symbol"] == "BTC/CAD"


def test_cross_exchange_fixed_fees_can_remove_a_gross_opportunity():
    data = {
        "ndax": _venue(0, {"XLM/CAD": ("XLM", "CAD", 0.99, 1.00)}),
        "kraken": _venue(0, {"XLM/CAD": ("XLM", "CAD", 1.02, 1.03)}),
    }

    assert find_cross_exchange_opportunities(
        data,
        starting_cad=1000,
        fixed_fee_cad_per_order=20,
        rebalance_cost_cad=0,
    ) == []


def test_cross_exchange_depth_walk_caps_size_at_profitable_book_levels():
    buy = _venue(0, {"XLM/CAD": ("XLM", "CAD", 0.99, 1.00)})
    sell = _venue(0, {"XLM/CAD": ("XLM", "CAD", 1.04, 1.05)})
    buy["order_books"] = {
        "XLM/CAD": {"asks": [[1.00, 50], [1.04, 100]], "bids": []}
    }
    sell["order_books"] = {
        "XLM/CAD": {"asks": [], "bids": [[1.05, 50], [1.02, 100]]}
    }

    opportunities = find_cross_exchange_opportunities(
        {"ndax": buy, "kraken": sell},
        starting_cad=200,
        fixed_fee_cad_per_order=0,
        rebalance_cost_cad=0,
    )

    assert len(opportunities) == 1
    result = opportunities[0]
    assert result["execution_basis"] == "order-book depth"
    assert result["base_quantity"] == pytest.approx(50)
    assert result["buy_spent_cad"] == pytest.approx(50)
    assert result["net_profit_cad"] == pytest.approx(2.5)
    assert len(result["depth_levels"]) == 1


def test_cross_exchange_depth_walk_accounts_for_fees_and_budget():
    buy = _venue(0.01, {"XLM/CAD": ("XLM", "CAD", 0.99, 1.00)})
    sell = _venue(0.01, {"XLM/CAD": ("XLM", "CAD", 1.04, 1.05)})
    buy["order_books"] = {"XLM/CAD": {"asks": [[1.00, 100]], "bids": []}}
    sell["order_books"] = {"XLM/CAD": {"asks": [], "bids": [[1.05, 100]]}}

    result = find_cross_exchange_opportunities(
        {"ndax": buy, "kraken": sell},
        starting_cad=100,
        fixed_fee_cad_per_order=0,
        rebalance_cost_cad=0,
    )[0]

    assert result["buy_spent_cad"] <= 100
    assert result["base_quantity"] == pytest.approx(100 / 1.01)
    assert result["net_profit_cad"] == pytest.approx(
        result["base_quantity"] * (1.05 * 0.99 - 1.00 * 1.01)
    )


def test_available_unprofitable_depth_does_not_fall_back_to_ticker_spread():
    buy = _venue(0, {"XLM/CAD": ("XLM", "CAD", 0.99, 1.00)})
    sell = _venue(0, {"XLM/CAD": ("XLM", "CAD", 1.04, 1.05)})
    buy["order_books"] = {"XLM/CAD": {"asks": [[1.10, 100]], "bids": []}}
    sell["order_books"] = {"XLM/CAD": {"asks": [], "bids": [[1.00, 100]]}}

    assert find_cross_exchange_opportunities(
        {"ndax": buy, "kraken": sell},
        starting_cad=100,
        fixed_fee_cad_per_order=0,
        rebalance_cost_cad=0,
    ) == []


def test_triangular_opportunity_accounts_for_fees_on_all_three_legs():
    venue = _venue(0.001, {
        "XLM/CAD": ("XLM", "CAD", 1.0, 1.01),
        "XRP/XLM": ("XRP", "XLM", 1.0, 1.01),
        "XRP/CAD": ("XRP", "CAD", 1.05, 1.06),
    })

    opportunities = find_triangular_opportunities(
        "ndax",
        venue,
        starting_cad=1000,
        fixed_fee_cad_per_order=0,
    )

    assert opportunities
    assert all(item["net_profit_cad"] > 0 for item in opportunities)
    assert any(item["route"] == "CAD -> XLM -> XRP -> CAD" for item in opportunities)


def test_triangular_scan_rejects_non_canadian_exchange():
    with pytest.raises(ValueError, match="Canadian arbitrage allowlist"):
        find_triangular_opportunities("binance", _venue(0, {}))


@pytest.mark.parametrize("fee", [-0.01, 1, 2])
def test_invalid_fee_rates_are_rejected(fee):
    with pytest.raises(ValueError, match="Taker fee"):
        validate_fee_rate(fee)


def test_non_spot_and_invalid_quotes_are_ignored():
    venue = _venue(0, {"XLM/CAD": ("XLM", "CAD", 1.0, 1.01)})
    venue["markets"]["XLM/CAD"]["spot"] = False

    assert find_triangular_opportunities("ndax", venue) == []


def test_snapshot_fetch_limits_tickers_and_order_books_to_pair_selection(monkeypatch):
    class FakeExchange:
        def __init__(self, exchange_id):
            self.exchange_id = exchange_id
            self.symbols_requested = []
            self.order_books_requested = []
        def load_markets(self):
            return {
                "XLM/CAD": {
                    "base": "XLM", "quote": "CAD", "spot": True, "active": True,
                },
                "XRP/XLM": {
                    "base": "XRP", "quote": "XLM", "spot": True, "active": True,
                },
                "BTC/USDT": {
                    "base": "BTC", "quote": "USDT", "spot": True, "active": True,
                },
                "XLM/USD": {
                    "base": "XLM", "quote": "USD", "spot": True, "active": True,
                },
                "USD/CAD": {
                    "base": "USD", "quote": "CAD", "spot": True, "active": True,
                },
                "XLM/CAD:CAD": {
                    "base": "XLM", "quote": "CAD", "spot": False, "active": True,
                },
            }

        def fetch_tickers(self, symbols):
            self.symbols_requested.extend(symbols)
            return {
                symbol: (
                    {"bid": 1.05, "ask": 1.06}
                    if self.exchange_id == "kraken" and symbol == "XLM/CAD"
                    else {"bid": 0.99, "ask": 1.00}
                )
                for symbol in symbols
            }

        def fetch_order_book(self, symbol, limit):
            self.order_books_requested.append((symbol, limit))
            return (
                {"bids": [], "asks": [[1.00, 100]]}
                if self.exchange_id == "ndax"
                else {"bids": [[1.05, 100]], "asks": []}
            )

    exchanges = {name: FakeExchange(name) for name in ("ndax", "kraken")}
    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(**{
        name: (lambda _options, exchange=exchange: exchange)
        for name, exchange in exchanges.items()
    }))

    snapshots, errors = fetch_canadian_spot_snapshots(("ndax",))

    assert not errors
    assert set(exchanges["ndax"].symbols_requested) == {
        "XLM/CAD", "XRP/XLM", "BTC/USDT", "XLM/USD", "USD/CAD",
    }
    assert snapshots["ndax"]["order_books"] == {}
    assert exchanges["ndax"].order_books_requested == []

    snapshots, errors = fetch_canadian_spot_snapshots(
        ("ndax", "kraken"), include_order_books=True
    )

    assert not errors
    assert set(exchanges["ndax"].order_books_requested) == {("XLM/CAD", 20)}
    assert set(exchanges["kraken"].order_books_requested) == {("XLM/CAD", 20)}
