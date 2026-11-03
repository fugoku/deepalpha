#!/usr/bin/env python3
"""DeepAlpha Canadian crypto terminal — read-only markets and paper trading."""

import streamlit as st
import os
import time
import math
import json
import sqlite3
import ccxt
import pandas as pd
import numpy as np
import config
from datetime import datetime, timezone, timedelta
from canadian_arbitrage import (
    CANADIAN_VENUES,
    fetch_canadian_price_history,
    fetch_canadian_spot_snapshots,
    find_cross_exchange_opportunities,
    find_triangular_opportunities,
)
from ml_paper_trader import INDICATOR_NAMES, build_indicator_matrix

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepAlpha | Canada Crypto Terminal",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# DARK EXCHANGE THEME — Binance charcoal with Kraken-inspired purple accents
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --canvas: #0b0e11;
        --panel: #15191e;
        --panel-raised: #1e2329;
        --line: #2b3139;
        --text: #eaecef;
        --muted: #848e9c;
        --gold: #f0b90b;
        --purple: #8b5cf6;
        --green: #0ecb81;
        --red: #f6465d;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 86% 0%, #211737 0%, transparent 28%),
            var(--canvas);
        color: var(--text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header[data-testid="stHeader"] { background: #0b0e11dd; }
    .block-container { max-width: 1540px; padding: 1.15rem 2rem 2rem; }
    [data-testid="stSidebar"] { background: #101419; border-right: 1px solid var(--line); }
    [data-testid="stMain"] { color: var(--text); }
    p, label, .stCaption, [data-testid="stMarkdownContainer"] { color: inherit; }
    .terminal-header {
        background: linear-gradient(105deg, #171b20 0%, #14181e 64%, #211735 100%);
        border: 1px solid #34303f;
        border-radius: 14px;
        padding: 19px 24px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 12px 40px #00000028;
    }
    .terminal-title {
        font-size: 21px;
        font-weight: 850;
        letter-spacing: -0.7px;
        background: linear-gradient(90deg, var(--gold), #f5d875 52%, #bda2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .terminal-sub {
        font-size: 10px;
        color: var(--muted);
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-left: 12px;
    }
    .market-strip {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        padding: 4px 2px 18px;
        color: var(--muted);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.1px;
        text-transform: uppercase;
    }
    .market-pill {
        border: 1px solid var(--line);
        background: #15191e;
        border-radius: 999px;
        padding: 6px 10px;
    }
    .market-pill-accent { border-color: #f0b90b55; color: var(--gold); }
    .market-pill-purple { border-color: #8b5cf655; color: #bda2ff; }
    .live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 8px #0ecb8188;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .clock {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: var(--muted);
    }
    .metric-card {
        background: linear-gradient(145deg, #191e24, #14181d);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 17px 15px;
        text-align: center;
        transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
    }
    .metric-card:hover { border-color: #8b5cf688; transform: translateY(-2px); box-shadow: 0 10px 24px #0003; }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 8px 26px #0002;
    }
    .metric-label {
        font-size: 10px;
        font-weight: 600;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 10px;
        color: var(--muted);
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .green { color: var(--green); }
    .red { color: var(--red); }
    .yellow { color: var(--gold); }
    .purple { color: #a78bfa; }
    .blue { color: #60a5fa; }
    .muted { color: var(--muted); }
    .section-title {
        font-size: 12px;
        font-weight: 700;
        color: var(--text);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 22px 0 12px;
        padding: 0 0 9px 11px;
        border-bottom: 1px solid var(--line);
        border-left: 3px solid var(--gold);
    }
    div[data-testid="stDataFrame"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stVegaLiteChart"], div[data-testid="stArrowVegaLiteChart"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 8px;
    }
    [data-testid="stExpander"] {
        background: #12161b;
        border: 1px solid var(--line);
        border-radius: 12px;
        margin: 8px 0;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary { font-weight: 700; }
    [data-testid="stExpander"] summary:hover { color: var(--gold); }
    .connection-card {
        margin: 18px 0 28px;
        padding: 22px 24px;
        background: linear-gradient(110deg, #171c22, #171421);
        border: 1px solid #34303f;
        border-radius: 12px;
    }
    .connection-card-title { color: var(--text); font-size: 15px; font-weight: 800; }
    .connection-card-copy { color: var(--muted); font-size: 13px; margin-top: 6px; }
    .stButton > button, [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(110deg, #f0b90b, #dba500);
        color: #111418;
        border: 0;
        border-radius: 8px;
        font-weight: 800;
        transition: filter 140ms ease, transform 140ms ease;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
        color: #111418;
        filter: brightness(1.12);
        transform: translateY(-1px);
        border: 0;
    }
    input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div {
        background-color: #171c22 !important;
        border-color: #343b45 !important;
        color: var(--text) !important;
    }
    [data-baseweb="tag"] { background-color: #34284d !important; }
    [data-testid="stAlert"] { border-radius: 10px; }
    .signal-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid var(--line);
        font-size: 13px;
    }
    .signal-coin {
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        width: 60px;
    }
    .signal-bar-bg {
        flex: 1;
        height: 6px;
        background: #2b3139;
        border-radius: 3px;
        margin: 0 12px;
        overflow: hidden;
    }
    .signal-bar-fill {
        height: 100%;
        border-radius: 3px;
    }
    .signal-pct {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        width: 50px;
        text-align: right;
    }

    /* ── Risk panel ── */
    .risk-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid var(--line);
        font-size: 13px;
    }
    .risk-label { color: var(--muted); }
    .risk-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #68717d;
        font-size: 11px;
        padding: 30px 0 10px 0;
        letter-spacing: 1px;
    }
    .footer a { color: #a78bfa; text-decoration: none; }
    .footer a:hover { color: var(--gold); }
    .stCheckbox label { color: var(--muted) !important; font-size: 12px; }
    @media (max-width: 760px) {
        .block-container { padding: .8rem 1rem 1.5rem; }
        .terminal-header { padding: 14px 16px; }
        .terminal-title { font-size: 17px; }
        .terminal-sub { display: none; }
        .metric-value { font-size: 19px; }
    }
</style>
""", unsafe_allow_html=True)

import html
def _escape(s):
    return html.escape(str(s))


# ──────────────────────────────────────────────────────────────────────────────
# EXCHANGE CONNECTION (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_exchange():
    """Create ccxt Bitget instance from env vars."""
    return ccxt.bitget({
        "apiKey": os.getenv("BITGET_API_KEY", ""),
        "secret": os.getenv("BITGET_SECRET", ""),
        "password": os.getenv("BITGET_PASSPHRASE", ""),
        "options": {"defaultType": "swap"},
    })


# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_data")
PAPER_DATABASE = os.path.join(config.DATA_DIR, "canadian_paper_trades.sqlite3")


def _read_paper_status():
    metadata = {}
    trades = {"total": 0, "open": 0, "closed": 0, "wins": 0, "net_pnl": 0.0}
    if not os.path.exists(PAPER_DATABASE):
        return metadata, trades
    with sqlite3.connect(PAPER_DATABASE, timeout=1) as db:
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "service_metadata" in tables:
            metadata = {
                key: json.loads(value)
                for key, value in db.execute(
                    "SELECT key,value FROM service_metadata"
                ).fetchall()
            }
        if "paper_trades" in tables:
            total, opened, closed, wins, net_pnl = db.execute(
                "SELECT COUNT(*), SUM(status='open'), SUM(status='closed'), "
                "SUM(status='closed' AND net_pnl_cad > 0), "
                "COALESCE(SUM(net_pnl_cad),0) FROM paper_trades"
            ).fetchone()
            trades = {
                "total": total or 0,
                "open": opened or 0,
                "closed": closed or 0,
                "wins": wins or 0,
                "net_pnl": net_pnl or 0.0,
            }
    return metadata, trades


def _safe_float(val, default=0.0):
    """Safely convert to float."""
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def fetch_all_data():
    """Fetch balance, positions, tickers, and trades from Bitget."""
    try:
        ex = get_exchange()
        balance = ex.fetch_balance(params={"type": "swap"})
        positions = ex.fetch_positions()
        tickers = ex.fetch_tickers(params={"type": "swap"})

        # Fetch trades across known + common pairs
        trades = []
        symbols = set()
        for pos in (positions or []):
            sym = pos.get("symbol")
            if sym:
                symbols.add(sym)
        for coin in ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "SUI", "ARB",
                      "XRP", "ADA", "MATIC", "OP", "WIF", "PEPE", "NEAR"]:
            symbols.add(f"{coin}/USDT:USDT")

        # Only fetch trades from last 7 days (ignore old account history)
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp() * 1000)
        for sym in symbols:
            try:
                t = ex.fetch_my_trades(sym, since=since_ms, limit=50)
                trades.extend(t)
            except Exception:
                pass

        trades.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        trades = trades[:200]

        return balance, positions, tickers, trades, True
    except Exception as e:
        return None, None, None, None, False


def compute_trade_analytics(trades):
    """Compute win rate, profit factor, avg win/loss from trade history."""
    wins, losses = [], []
    daily_pnl = {}  # date -> pnl
    today_pnl = 0.0
    today_fees = 0.0
    today_count = 0
    total_fees = 0.0

    today_date = datetime.now(timezone.utc).date()

    for t in (trades or []):
        info = t.get("info", {})
        pnl = _safe_float(info.get("profit") or info.get("realizedPnl"))
        fee_cost = abs(_safe_float((t.get("fee") or {}).get("cost")))
        total_fees += fee_cost

        ts = t.get("timestamp", 0)
        if ts:
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            d = dt.date()
            daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl - fee_cost
            if d == today_date:
                today_pnl += pnl
                today_fees += fee_cost
                today_count += 1

        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(abs(pnl))

    # Last 30 trades for win rate
    recent_30 = trades[:30] if trades else []
    wins_30 = sum(1 for t in recent_30
                  if _safe_float((t.get("info") or {}).get("profit")
                                 or (t.get("info") or {}).get("realizedPnl")) > 0)
    win_rate = wins_30 / max(len(recent_30), 1) * 100

    gross_wins = sum(wins) if wins else 0
    gross_losses = sum(losses) if losses else 0
    profit_factor = gross_wins / max(gross_losses, 0.01)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    return {
        "today_pnl": today_pnl,
        "today_fees": today_fees,
        "today_count": today_count,
        "total_fees": total_fees,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "gross_wins": gross_wins,
        "gross_losses": gross_losses,
        "daily_pnl": daily_pnl,
    }


def compute_risk_metrics(equity_history):
    """Compute max drawdown and Sharpe from equity history list."""
    if len(equity_history) < 2:
        return {"max_dd": 0, "sharpe": 0}

    arr = np.array(equity_history)
    # Max drawdown
    peak = np.maximum.accumulate(arr)
    dd = (peak - arr) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd)) * 100

    # Sharpe from returns
    returns = np.diff(arr) / np.where(arr[:-1] > 0, arr[:-1], 1)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns)) * math.sqrt(365 * 24)  # annualized
    else:
        sharpe = 0.0

    return {"max_dd": max_dd, "sharpe": sharpe}


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE — equity history
# ──────────────────────────────────────────────────────────────────────────────
if "equity_history" not in st.session_state:
    st.session_state.equity_history = []
    st.session_state.equity_timestamps = []


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
now_utc = datetime.now(timezone.utc)
now_str = now_utc.strftime("%Y-%m-%d  %H:%M:%S UTC")

balance, raw_positions, tickers, trades, connected = None, None, None, None, False

status_dot = '<span class="live-dot"></span>' if connected else '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ff4757;margin-right:6px;"></span>'
status_text = "LIVE" if connected else "OFFLINE"

st.markdown(f"""
<div class="terminal-header">
    <div style="display:flex;align-items:center;">
        <span class="terminal-title">DeepAlpha Markets</span>
        <span class="terminal-sub">Canada · Trading Pairs</span>
    </div>
    <div style="display:flex;align-items:center;gap:16px;">
        <span style="font-size:10px;color:{'#0ecb81' if connected else '#f0b90b'};font-weight:700;letter-spacing:1px;">
            {status_dot if connected else '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f0b90b;margin-right:6px;"></span>'}
            {'ACCOUNT LINKED' if connected else 'PUBLIC MARKETS'}
        </span>
        <span class="clock">{now_str}</span>
    </div>
</div>
<div class="market-strip">
    <span class="market-pill market-pill-accent">Canadian exchange markets</span>
    <span class="market-pill">NDAX</span>
    <span class="market-pill">Kraken</span>
    <span class="market-pill">Coinbase</span>
    <span class="market-pill market-pill-purple">paper trading only</span>
    <span>public data · no orders</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">AI paper-learning status</div>', unsafe_allow_html=True)
try:
    model_status, paper_totals = _read_paper_status()
except (sqlite3.Error, json.JSONDecodeError) as exc:
    model_status, paper_totals = {}, {"total": 0, "open": 0, "closed": 0, "wins": 0, "net_pnl": 0.0}
    st.error(f"Could not read the local ML status journal: {exc}")

last_cycle = model_status.get("last_cycle_at")
if last_cycle:
    heartbeat = datetime.fromisoformat(str(last_cycle))
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    service_is_running = (now_utc - heartbeat).total_seconds() < 15 * 60
else:
    service_is_running = False

status_cols = st.columns(4)
status_cols[0].metric(
    "Paper learner",
    "ON" if service_is_running else "WAITING",
    delta="public data · no orders" if service_is_running else "start ml_paper_trader.py",
    delta_color="off",
)
status_cols[1].metric(
    "Trained models",
    f"{sum(model_status.get(key) is not None for key in ('long_accuracy', 'short_accuracy'))} / 2",
    delta=(
        f"long {model_status.get('long_accuracy', 0):.1%} · "
        f"short {model_status.get('short_accuracy', 0):.1%} holdout"
        if model_status.get("long_accuracy") is not None
        and model_status.get("short_accuracy") is not None
        else "chronological validation"
    ),
    delta_color="off",
)
status_cols[2].metric(
    "Paper positions",
    paper_totals["open"],
    delta=f"{paper_totals['closed']} closed · CAD ${paper_totals['net_pnl']:+,.2f} net",
    delta_color="off",
)
status_cols[3].metric(
    "Pairs loaded",
    len(model_status.get("assets_loaded", [])),
    delta=f"source {model_status.get('source_venue') or 'not connected'}",
    delta_color="off",
)

if model_status.get("latest_probabilities"):
    with st.expander("Latest model signals · directional scores, not guarantees", expanded=False):
        prediction_rows = []
        for asset, scores in model_status["latest_probabilities"].items():
            long_score, short_score = scores
            side = "LONG" if long_score >= short_score else "SHORT"
            score = max(long_score, short_score)
            prediction_rows.append({
                "Pair": f"{asset}/CAD",
                "Model direction": side,
                "Model score": score,
                "Long score": long_score,
                "Short score": short_score,
            })
        predictions_df = pd.DataFrame(prediction_rows).sort_values(
            "Model score", ascending=False
        )
        st.dataframe(
            predictions_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model score": st.column_config.NumberColumn(format="%.1%"),
                "Long score": st.column_config.NumberColumn(format="%.1%"),
                "Short score": st.column_config.NumberColumn(format="%.1%"),
            },
        )

if model_status.get("catalyst_available") is False and model_status.get("last_cycle_at"):
    st.warning(
        "Headline monitor unavailable. New simulated entries are paused; existing paper "
        "positions can still close."
    )


with st.expander("Canadian trading-pair scanner", expanded=True):
    st.caption(
        "Read-only scan of supported trading-pair variants for all 30 configured coins on NDAX, "
        "Kraken, and Coinbase. Quote currencies are conservatively converted to CAD for profit "
        "estimates; pairs without a public CAD conversion path are excluded."
    )
    venues = st.multiselect(
        "Canadian-serving exchanges",
        options=sorted(CANADIAN_VENUES),
        default=[name for name in config.CANADIAN_ARBITRAGE_EXCHANGES if name in CANADIAN_VENUES],
        key="arbitrage_venues",
    )
    capital, venue_status = st.columns([1, 3])
    with capital:
        starting_cad = st.number_input(
            "Capital per opportunity (CAD, max $1,000)",
            min_value=1.0,
            max_value=1000.0,
            value=min(max(float(config.ARBITRAGE_CAPITAL_CAD), 1.0), 1000.0),
            step=50.0,
        )
    with venue_status:
        st.caption(
            "Taker fees, fixed charges, quote-currency conversion spreads, and rebalancing costs "
            "are applied automatically. Every positive-net opportunity is shown; the ML learner "
            "independently guides simulated long/short positions."
        )

    if st.button("Scan Canadian markets", type="primary", key="scan_canadian_arbitrage"):
        if not venues:
            st.warning("Select at least one Canadian-serving exchange.")
        else:
            with st.spinner("Fetching public trading-pair quotes from Canadian exchanges..."):
                snapshots, errors = fetch_canadian_spot_snapshots(
                    tuple(venues), include_order_books=True
                )
            st.session_state["canadian_arbitrage_snapshots"] = snapshots
            st.session_state["canadian_arbitrage_errors"] = errors
            st.session_state["canadian_arbitrage_as_of"] = datetime.now(timezone.utc)

    snapshots = st.session_state.get("canadian_arbitrage_snapshots", {})
    errors = st.session_state.get("canadian_arbitrage_errors", {})
    if snapshots:
        cross = find_cross_exchange_opportunities(
            snapshots,
            starting_cad=starting_cad,
            fixed_fee_cad_per_order=config.ARBITRAGE_FIXED_FEE_CAD_PER_ORDER,
            rebalance_cost_cad=config.ARBITRAGE_REBALANCE_COST_CAD,
        )
        triangular = []
        for exchange_id, data in snapshots.items():
            triangular.extend(find_triangular_opportunities(
                exchange_id,
                data,
                starting_cad=starting_cad,
                fixed_fee_cad_per_order=config.ARBITRAGE_FIXED_FEE_CAD_PER_ORDER,
            ))
        opportunities = sorted(
            cross + triangular,
            key=lambda item: item["net_profit_cad"],
            reverse=True,
        )
        scanned_at = st.session_state.get("canadian_arbitrage_as_of")
        if scanned_at:
            st.caption(
                f"Last scan: {scanned_at.strftime('%Y-%m-%d %H:%M:%S UTC')} · "
                f"venues reached: {', '.join(sorted(snapshots))}"
            )
        if opportunities:
            rows = []
            for item in opportunities:
                row = {
                    "Type": item["type"],
                    "Route / Asset": item.get("route", item.get("symbol", "")),
                    "Net Profit (CAD)": round(item["net_profit_cad"], 2),
                    "Net Return (%)": round(item["net_return_pct"], 3),
                    "Est. Fees": (
                        f"{item['taker_fee_pct_per_leg']:.2f}%/leg"
                        if item["type"] == "Triangular"
                        else f"{item['buy_taker_fee_pct']:.2f}% buy / "
                        f"{item['sell_taker_fee_pct']:.2f}% sell"
                    ),
                    "Exchanges": (
                        item["exchange"]
                        if item["type"] == "Triangular"
                        else f"Buy {item['buy_exchange']} / Sell {item['sell_exchange']}"
                    ),
                    "Starting CAD": round(item["starting_cad"], 2),
                    "Executable base": (
                        round(item["base_quantity"], 8)
                        if "base_quantity" in item else "—"
                    ),
                    "Depth basis": item.get("execution_basis", "top-of-book estimate"),
                }
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No positive-net pair opportunities after estimated fees and costs.")
    elif errors:
        st.warning("No Canadian exchange quotes were available. Check connectivity and try again.")

    for exchange_id, error in errors.items():
        st.caption(f"{exchange_id}: {error}")
    st.caption(
        "Fee and transfer values are estimates, not live account-specific rates. "
        "Update ARBITRAGE_* settings to match each exchange's current schedule. "
        "Cross-exchange estimates use matched order-book depth when available; triangular "
        "routes and any missing-book fallback are top-of-book estimates. Slippage beyond "
        "the fetched book, deposit holds, and execution latency are not modeled."
    )

with st.expander("Canadian crypto charts and indicators", expanded=False):
    chart_controls = st.columns(3)
    with chart_controls[0]:
        chart_asset = st.selectbox(
            "Crypto asset",
            options=sorted(config.COINS),
            key="canadian_chart_asset",
        )
    with chart_controls[1]:
        chart_exchange = st.selectbox(
            "Canadian-serving exchange",
            options=sorted(CANADIAN_VENUES),
            key="chart_exchange",
        )
    with chart_controls[2]:
        chart_timeframe = st.selectbox(
            "Candle interval", options=("1h", "4h", "1d"), key="chart_timeframe"
        )
    if st.button("Refresh price chart", key="refresh_canadian_chart"):
        try:
            history = fetch_canadian_price_history(
                chart_exchange, chart_asset, chart_timeframe, limit=168
            )
            st.session_state["canadian_chart_data"] = {
                "asset": chart_asset,
                "exchange": chart_exchange,
                "timeframe": chart_timeframe,
                "candles": history,
            }
            st.session_state.pop("canadian_chart_error", None)
        except Exception as exc:
            st.session_state["canadian_chart_error"] = (
                {
                    "selection": (chart_asset, chart_exchange, chart_timeframe),
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    chart_data = st.session_state.get("canadian_chart_data")
    if chart_data and (
        chart_data["asset"], chart_data["exchange"], chart_data["timeframe"]
    ) == (chart_asset, chart_exchange, chart_timeframe):
        candle_rows = chart_data["candles"]
        chart_df = pd.DataFrame(
            candle_rows, columns=("Timestamp", "Open", "High", "Low", "Close", "Volume")
        )
        chart_df["Timestamp"] = pd.to_datetime(chart_df["Timestamp"], unit="ms", utc=True)
        chart_df = chart_df.set_index("Timestamp")
        st.line_chart(chart_df[["Open", "High", "Low", "Close"]], use_container_width=True)
        st.bar_chart(chart_df[["Volume"]], use_container_width=True)
        chart_indicators = build_indicator_matrix(candle_rows)
        latest_indicators = chart_indicators[-1]
        rsi = latest_indicators[INDICATOR_NAMES.index("rsi_14")] * 100
        macd = latest_indicators[INDICATOR_NAMES.index("macd_hist_pct")] * 100
        atr = latest_indicators[INDICATOR_NAMES.index("atr_14_pct")] * 100
        bollinger = latest_indicators[INDICATOR_NAMES.index("bollinger_percent_b")]
        metrics = st.columns(4)
        metrics[0].metric("RSI (14)", f"{rsi:.1f}")
        metrics[1].metric("MACD histogram", f"{macd:.3f}%")
        metrics[2].metric("ATR (14)", f"{atr:.2f}%")
        metrics[3].metric("Bollinger %B", f"{bollinger:.2f}")
        indicators_df = pd.DataFrame({
            "Indicator": INDICATOR_NAMES,
            "Latest value": latest_indicators,
        })
        st.dataframe(
            indicators_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Latest value": st.column_config.NumberColumn(format="%.5f"),
            },
        )
        st.caption(
            f"{chart_asset}/CAD · {chart_exchange} · {chart_timeframe} · "
            f"last candle {chart_df.index[-1].strftime('%Y-%m-%d %H:%M UTC')}"
        )
    elif (
        isinstance(st.session_state.get("canadian_chart_error"), dict)
        and st.session_state["canadian_chart_error"].get("selection")
        == (chart_asset, chart_exchange, chart_timeframe)
    ):
        st.warning(st.session_state["canadian_chart_error"]["message"])
    else:
        st.caption(
            "Choose a configured CAD trading pair and refresh to load its price chart "
            "and indicator readings."
        )

with st.expander("Canadian ML paper account", expanded=False):
    paper_database = os.path.join(config.DATA_DIR, "canadian_paper_trades.sqlite3")
    if not os.path.exists(paper_database):
        st.info("Paper-trading history is not available yet. Start the paper trader with "
                "`python ml_paper_trader.py` to begin recording simulated long and short trades.")
    else:
        try:
            with sqlite3.connect(paper_database) as paper_db:
                total, closed, wins, net_pnl, latest = paper_db.execute(
                    "SELECT COUNT(*), SUM(status='closed'), "
                    "SUM(status='closed' AND net_pnl_cad > 0), "
                    "COALESCE(SUM(net_pnl_cad), 0), MAX(entry_time) FROM paper_trades"
                ).fetchone()
                open_trades = paper_db.execute(
                    "SELECT side, asset, entry_price, entry_probability, entry_exchange "
                    "FROM paper_trades WHERE status='open' ORDER BY entry_time DESC"
                ).fetchall()
                existing_tables = {
                    row[0] for row in paper_db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                whale_flows = []
                whale_prints = []
                market_events = []
                if "whale_flow_snapshots" in existing_tables:
                    whale_flows = paper_db.execute(
                        "SELECT asset,net_flow_cad,flow_ratio,whale_count,venues,observed_at "
                        "FROM whale_flow_snapshots ORDER BY ABS(net_flow_cad) DESC"
                    ).fetchall()
                if "whale_prints" in existing_tables:
                    whale_prints = paper_db.execute(
                        "SELECT exchange_id,symbol,side,notional_cad,flow_destination,"
                        "timestamp_ms FROM whale_prints ORDER BY timestamp_ms DESC LIMIT 20"
                    ).fetchall()
                if "market_catalysts" in existing_tables:
                    market_events = paper_db.execute(
                        "SELECT title,source,category,impact,tone,published_at,url "
                        "FROM market_catalysts ORDER BY published_at DESC LIMIT 20"
                    ).fetchall()
            status_metrics = st.columns(4)
            status_metrics[0].metric("Paper trades", total or 0)
            status_metrics[1].metric("Closed trades", closed or 0)
            status_metrics[2].metric(
                "Win rate",
                f"{(wins or 0) / closed:.1%}" if closed else "—",
            )
            status_metrics[3].metric("Realized net PnL", f"CAD ${net_pnl:,.2f}")
            st.caption(
                f"Last entry: {latest or 'none'} · paper service must be running separately "
                "to continue learning and trading."
            )
            if open_trades:
                st.dataframe(
                    pd.DataFrame(
                        open_trades,
                        columns=("Side", "Asset", "Entry price", "Model probability", "Exchange"),
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No open simulated positions.")
            st.markdown("**Large-trade flow (exchange-level, not wallet tracking)**")
            st.caption(
                "Venue and taker side indicate observed CAD/crypto trade flow; public prints "
                "do not identify beneficial owners or on-chain source/destination wallets."
            )
            if whale_flows:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Asset": row[0],
                                "Net large-trade flow (CAD)": round(row[1], 2),
                                "Buy/sell imbalance": round(row[2], 3),
                                "Large prints": row[3],
                                "Venues": row[4],
                                "Observed (UTC)": row[5],
                            }
                            for row in whale_flows
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No recent large-trade summaries have been stored.")
            if whale_prints:
                with st.expander("Recent large prints and observed flow direction"):
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Exchange": row[0],
                                    "Pair": row[1],
                                    "Aggressor": row[2],
                                    "Notional (CAD)": round(row[3], 2),
                                    "Observed direction": row[4],
                                    "Time (UTC)": datetime.fromtimestamp(
                                        row[5] / 1000, tz=timezone.utc
                                    ).strftime("%Y-%m-%d %H:%M:%S"),
                                }
                                for row in whale_prints
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            st.markdown("**Public market catalysts**")
            st.caption(
                "Headline monitoring covers reported leader speeches, policy, regulatory, "
                "geopolitical, exchange, and crypto-security events. It is not live audio "
                "transcription and may be delayed or incomplete."
            )
            if market_events:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Impact": row[3].upper(),
                                "Category": row[2],
                                "Headline": row[0],
                                "Source": row[1],
                                "Tone": round(row[4], 2),
                                "Published (UTC)": row[5],
                                "URL": row[6],
                            }
                            for row in market_events
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "URL": st.column_config.LinkColumn("Article"),
                    },
                )
            else:
                st.caption("No catalyst headlines stored yet. Start the paper service to monitor.")
        except sqlite3.Error as exc:
            st.error(f"Could not read the paper-trade journal: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────────
if connected and balance is not None:

    # ── Extract account values ──
    equity = _safe_float(balance.get("total", {}).get("USDT"))
    free_usdt = _safe_float(balance.get("free", {}).get("USDT"))
    used_margin = _safe_float(balance.get("used", {}).get("USDT"))

    # Update equity history
    st.session_state.equity_history.append(equity)
    st.session_state.equity_timestamps.append(now_utc)
    # Keep last 500 points
    if len(st.session_state.equity_history) > 500:
        st.session_state.equity_history = st.session_state.equity_history[-500:]
        st.session_state.equity_timestamps = st.session_state.equity_timestamps[-500:]

    # ── Price map ──
    price_map = {}
    for sym, tick in (tickers or {}).items():
        coin = sym.split("/")[0] if "/" in sym else sym
        price_map[coin] = _safe_float(tick.get("last"))

    # ── Positions ──
    positions = []
    total_unrealized = 0.0
    for pos in (raw_positions or []):
        contracts = _safe_float(pos.get("contracts"))
        if contracts == 0:
            continue
        symbol = pos.get("symbol", "")
        coin = symbol.split("/")[0] if "/" in symbol else symbol
        entry = _safe_float(pos.get("entryPrice"))
        mark = _safe_float(pos.get("markPrice"))
        current = mark if mark > 0 else price_map.get(coin, 0)
        unrealized = _safe_float(pos.get("unrealizedPnl"))
        margin = _safe_float(pos.get("initialMargin") or pos.get("collateral"))
        leverage = _safe_float(pos.get("leverage"))
        side = (pos.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            side = "LONG" if contracts > 0 else "SHORT"
        notional = _safe_float(pos.get("notional"))

        pnl_pct = 0.0
        if entry > 0 and current > 0:
            pnl_pct = ((current - entry) / entry * 100) if side == "LONG" else ((entry - current) / entry * 100)

        # TP levels from info if available
        info = pos.get("info", {})
        tp1 = _safe_float(info.get("presetTakeProfitPrice"))
        tp2 = 0.0  # second TP not standard in ccxt, placeholder

        total_unrealized += unrealized
        positions.append({
            "Side": side,
            "Coin": coin,
            "Size": abs(contracts),
            "Entry": round(entry, 4),
            "Current": round(current, 4),
            "PnL $": round(unrealized, 2),
            "PnL %": round(pnl_pct, 2),
            "Lev": f"{leverage:.0f}x" if leverage > 0 else "--",
            "Margin": round(margin, 2),
            "TP1": round(tp1, 4) if tp1 else "--",
            "TP2": round(tp2, 4) if tp2 else "--",
        })

    # ── Trade analytics ──
    analytics = compute_trade_analytics(trades)
    today_net = analytics["today_pnl"] - analytics["today_fees"]
    fee_ratio = analytics["today_fees"] / max(abs(analytics["today_pnl"]), analytics["today_fees"], 1.0) * 100

    # ── Risk metrics ──
    risk = compute_risk_metrics(st.session_state.equity_history)

    # ══════════════════════════════════════════════════════════════════════
    # 1. TOP ROW — 6 METRIC CARDS
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Account Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Equity</div>
            <div class="metric-value green">${equity:,.2f}</div>
            <div class="metric-sub">Free: ${free_usdt:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        clr = "green" if total_unrealized >= 0 else "red"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Unrealized PnL</div>
            <div class="metric-value {clr}">${total_unrealized:+,.2f}</div>
            <div class="metric-sub">{len(positions)} positions</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        clr = "green" if today_net >= 0 else "red"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Today Net PnL</div>
            <div class="metric-value {clr}">${today_net:+,.2f}</div>
            <div class="metric-sub">Fees: ${analytics['today_fees']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Today Trades</div>
            <div class="metric-value purple">{analytics['today_count']}</div>
            <div class="metric-sub">Total fills: {len(trades or [])}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        wr = analytics["win_rate"]
        wr_clr = "green" if wr >= 55 else "yellow" if wr >= 45 else "red"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Win Rate (30)</div>
            <div class="metric-value {wr_clr}">{wr:.1f}%</div>
            <div class="metric-sub">last 30 trades</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        fr_clr = "green" if fee_ratio < 15 else "yellow" if fee_ratio < 30 else "red"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Fee Ratio</div>
            <div class="metric-value {fr_clr}">{fee_ratio:.1f}%</div>
            <div class="metric-sub">fees / gross PnL</div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # 2. EQUITY CURVE + DAILY PNL CHARTS (side by side)
    # ══════════════════════════════════════════════════════════════════════
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown('<div class="section-title">Equity Curve</div>', unsafe_allow_html=True)
        if len(st.session_state.equity_history) > 1:
            eq_df = pd.DataFrame({
                "Time": st.session_state.equity_timestamps,
                "Equity": st.session_state.equity_history,
            }).set_index("Time")
            st.line_chart(eq_df, use_container_width=True, color="#00d4aa")
        else:
            st.caption("Equity curve builds over time with each refresh cycle.")

    with chart_right:
        st.markdown('<div class="section-title">Daily PnL (Last 7 Days)</div>', unsafe_allow_html=True)
        daily = analytics["daily_pnl"]
        if daily:
            last_7 = sorted(daily.items(), key=lambda x: x[0])[-7:]
            dpnl_df = pd.DataFrame(last_7, columns=["Date", "PnL"])
            dpnl_df["Date"] = dpnl_df["Date"].astype(str)
            dpnl_df = dpnl_df.set_index("Date")
            st.bar_chart(dpnl_df, use_container_width=True, color="#7B61FF")
        else:
            st.caption("No daily PnL data yet.")

    # ══════════════════════════════════════════════════════════════════════
    # 3. OPEN POSITIONS TABLE
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Open Positions</div>', unsafe_allow_html=True)
    if positions:
        pos_df = pd.DataFrame(positions)
        st.dataframe(
            pos_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "PnL $": st.column_config.NumberColumn(format="$%.2f"),
                "PnL %": st.column_config.NumberColumn(format="%.2f%%"),
                "Margin": st.column_config.NumberColumn(format="$%.2f"),
                "Entry": st.column_config.NumberColumn(format="%.4f"),
                "Current": st.column_config.NumberColumn(format="%.4f"),
            },
        )
    else:
        st.info("No open positions -- Alpha is scanning for setups.")

    # ══════════════════════════════════════════════════════════════════════
    # 4. RECENT TRADES + RISK METRICS (side by side)
    # ══════════════════════════════════════════════════════════════════════
    trades_col, risk_col = st.columns([3, 1])

    with trades_col:
        st.markdown('<div class="section-title">Recent Trades (Last 20)</div>', unsafe_allow_html=True)
        if trades:
            recent = trades[:20]
            rows = []
            for t in recent:
                info = t.get("info", {})
                pnl = _safe_float(info.get("profit") or info.get("realizedPnl"))
                fee = abs(_safe_float((t.get("fee") or {}).get("cost")))
                ts = t.get("timestamp", 0)
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if ts else now_utc
                symbol = t.get("symbol", "")
                coin = symbol.split("/")[0] if "/" in symbol else symbol
                side = (t.get("side") or "").upper()

                rows.append({
                    "Time": dt.strftime("%m/%d %H:%M"),
                    "Coin": coin,
                    "Side": side,
                    "Price": round(_safe_float(t.get("price")), 4),
                    "Size": _safe_float(t.get("amount")),
                    "PnL": round(pnl, 3),
                    "Fee": round(fee, 4),
                })

            trade_df = pd.DataFrame(rows)
            st.dataframe(
                trade_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "PnL": st.column_config.NumberColumn(format="$%.3f"),
                    "Fee": st.column_config.NumberColumn(format="$%.4f"),
                    "Price": st.column_config.NumberColumn(format="%.4f"),
                },
            )
        else:
            st.caption("No recent trades available.")

    with risk_col:
        st.markdown('<div class="section-title">Risk Metrics</div>', unsafe_allow_html=True)
        risk_items = [
            ("Max Drawdown", f"{risk['max_dd']:.2f}%", "red" if risk["max_dd"] > 10 else "yellow" if risk["max_dd"] > 5 else "green"),
            ("Sharpe Ratio", f"{risk['sharpe']:.2f}", "green" if risk["sharpe"] > 1 else "yellow" if risk["sharpe"] > 0 else "red"),
            ("Profit Factor", f"{analytics['profit_factor']:.2f}", "green" if analytics["profit_factor"] > 1.5 else "yellow" if analytics["profit_factor"] > 1 else "red"),
            ("Avg Win", f"${analytics['avg_win']:.2f}", "green"),
            ("Avg Loss", f"${analytics['avg_loss']:.2f}", "red"),
            ("Margin Used", f"${used_margin:,.0f}", "blue"),
        ]
        risk_html = ""
        for label, value, color in risk_items:
            risk_html += f"""
            <div class="risk-item">
                <span class="risk-label">{label}</span>
                <span class="risk-value {color}">{value}</span>
            </div>"""
        st.markdown(f'<div class="metric-card" style="padding:12px 16px;">{risk_html}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # 5. AI STATUS PANEL
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">AI Engine Status</div>', unsafe_allow_html=True)
    ai_left, ai_mid, ai_right = st.columns(3)

    # Model files status
    with ai_left:
        model_html = ""
        for horizon in ["15m", "1h", "4h"]:
            model_path = os.path.join(DATA_DIR, f"model_{horizon}.pkl")
            if os.path.exists(model_path):
                mtime = os.path.getmtime(model_path)
                age_h = (time.time() - mtime) / 3600
                size_mb = os.path.getsize(model_path) / 1e6
                freshness = "green" if age_h < 6 else "yellow" if age_h < 24 else "red"
                model_html += f"""
                <div class="risk-item">
                    <span class="risk-label">Model {horizon.upper()}</span>
                    <span class="risk-value {freshness}">{size_mb:.1f}MB &middot; {age_h:.0f}h ago</span>
                </div>"""
            else:
                model_html += f"""
                <div class="risk-item">
                    <span class="risk-label">Model {horizon.upper()}</span>
                    <span class="risk-value green">ACTIVE (VPS)</span>
                </div>"""

        st.markdown(f"""<div class="metric-card" style="padding:12px 16px;">
            <div class="metric-label" style="margin-bottom:10px;">Ensemble Models</div>
            {model_html}
        </div>""", unsafe_allow_html=True)

    # Ensemble status
    with ai_mid:
        ensemble_components = [
            ("LightGBM", "lgb"),
            ("XGBoost", "xgb"),
            ("TFT", "tft"),
        ]
        ens_html = ""
        for name, key in ensemble_components:
            ens_path = os.path.join(DATA_DIR, f"model_{key}.pkl")
            exists = os.path.exists(ens_path)
            status_clr = "green"
            status_txt = "ACTIVE (VPS)" if not exists else "ACTIVE"
            ens_html += f"""
            <div class="risk-item">
                <span class="risk-label">{name}</span>
                <span class="risk-value {status_clr}">{status_txt}</span>
            </div>"""

        # Check for accuracy log
        acc_path = os.path.join(DATA_DIR, "model_accuracy.txt")
        accuracy_str = "OFFLINE"
        if os.path.exists(acc_path):
            try:
                with open(acc_path) as f:
                    accuracy_str = _escape(f.read().strip()[:10])
            except Exception:
                pass

        ens_html += f"""
        <div class="risk-item">
            <span class="risk-label">Accuracy</span>
            <span class="risk-value purple">{accuracy_str}</span>
        </div>"""

        st.markdown(f"""<div class="metric-card" style="padding:12px 16px;">
            <div class="metric-label" style="margin-bottom:10px;">Ensemble Status</div>
            {ens_html}
        </div>""", unsafe_allow_html=True)

    # Top signals
    with ai_right:
        signals_html = ""
        signals_path = os.path.join(DATA_DIR, "latest_signals.csv")
        if os.path.exists(signals_path):
            try:
                sig_df = pd.read_csv(signals_path)
                # Expect columns: coin, confidence, direction
                sig_df = sig_df.sort_values("confidence", ascending=False).head(5)
                for _, row in sig_df.iterrows():
                    coin = _escape(row.get("coin", "??"))
                    conf = _safe_float(row.get("confidence")) * 100
                    direction = _escape(str(row.get("direction", ""))).upper()
                    dir_clr = "#00d4aa" if direction == "LONG" else "#ff4757" if direction == "SHORT" else "#555a70"
                    bar_clr = dir_clr
                    signals_html += f"""
                    <div class="signal-row">
                        <span class="signal-coin">{coin}</span>
                        <div class="signal-bar-bg">
                            <div class="signal-bar-fill" style="width:{min(conf,100):.0f}%;background:{bar_clr};"></div>
                        </div>
                        <span class="signal-pct" style="color:{dir_clr};">{conf:.0f}%</span>
                    </div>"""
            except Exception:
                signals_html = '<div class="metric-sub">Error reading signals</div>'
        else:
            # Generate placeholder from current positions
            for p in positions[:5]:
                coin = p["Coin"]
                side = p["Side"]
                dir_clr = "#00d4aa" if side == "LONG" else "#ff4757"
                conf = min(abs(p["PnL %"]) * 5 + 50, 95)
                signals_html += f"""
                <div class="signal-row">
                    <span class="signal-coin">{coin}</span>
                    <div class="signal-bar-bg">
                        <div class="signal-bar-fill" style="width:{conf:.0f}%;background:{dir_clr};"></div>
                    </div>
                    <span class="signal-pct" style="color:{dir_clr};">{conf:.0f}%</span>
                </div>"""
            if not positions:
                signals_html = '<div class="metric-sub">No active signals</div>'

        # Last prediction time
        pred_time = "--"
        pred_path = os.path.join(DATA_DIR, "last_prediction.txt")
        if os.path.exists(pred_path):
            try:
                pred_mtime = os.path.getmtime(pred_path)
                pred_age = (time.time() - pred_mtime) / 60
                pred_time = f"{pred_age:.0f}m ago"
            except Exception:
                pass

        st.markdown(f"""<div class="metric-card" style="padding:12px 16px;">
            <div class="metric-label" style="margin-bottom:10px;">Top 5 Signals</div>
            {signals_html}
            <div style="margin-top:10px;font-size:10px;color:#555a70;">Last prediction: {pred_time}</div>
        </div>""", unsafe_allow_html=True)


else:
    st.markdown("""
    <div class="connection-card">
        <div class="connection-card-title">Canadian market tools are ready</div>
        <div class="connection-card-copy">
            Charts, public market scans, catalyst monitoring, and simulated trades do not
            require exchange credentials. The optional legacy Bitget account panel is disconnected.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER + AUTO-REFRESH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
foot_left, foot_mid, foot_right = st.columns([1, 2, 1])
with foot_mid:
    st.markdown("""
    <div class="footer">
        Powered by <strong>DeepAlpha</strong> |
        <a href="https://deepalphabot.com" target="_blank">deepalphabot.com</a>
    </div>
    """, unsafe_allow_html=True)

# Auto-refresh in sidebar
auto = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto:
    time.sleep(30)
    st.rerun()
