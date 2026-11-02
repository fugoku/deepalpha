# Configuration

All configuration is managed through environment variables loaded from a `.env` file. Default values are defined in `config.py`.

---

## Credentials

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIVATE_KEY` | `""` | Hyperliquid API private key (required) |
| `WALLET_ADDRESS` | `""` | Your Hyperliquid wallet address (required) |
| `TELEGRAM_TOKEN` | `""` | Telegram bot token for notifications (optional) |
| `TELEGRAM_CHAT_ID` | `""` | Telegram chat ID to receive notifications (optional) |

---

## Trading Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `LEVERAGE` | `5` | Leverage multiplier applied to all positions. Higher leverage amplifies both gains and losses. |
| `MAX_POSITIONS` | `3` | Maximum number of simultaneous open positions. Limits total exposure. |
| `RISK_PER_TRADE` | `0.10` | Fraction of equity allocated per trade (before leverage). At 10% with 5x leverage, each trade uses 50% of equity in notional. |
| `MAX_DAILY_LOSS_PCT` | `0.05` | Maximum daily loss as a fraction of equity. Trading pauses if this threshold is hit. |

---

## Risk Management

| Variable | Default | Description |
|----------|---------|-------------|
| `STOP_LOSS_PCT` | `0.02` | Fixed stop-loss distance from entry price (2%). A long entered at $100 triggers SL at $98. |
| `TAKE_PROFIT_PCT` | `0.03` | Fixed take-profit distance from entry price (3%). A long entered at $100 triggers TP at $103. |
| `CIRCUIT_BREAKER_LOSSES` | `3` | Number of consecutive losing trades before the circuit breaker activates. |
| `CIRCUIT_BREAKER_COOLDOWN` | `3600` | Cooldown period in seconds (1 hour) after the circuit breaker triggers. No new trades during this time. |

---

## AI Model

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `model.pkl` | File path to the trained model. Updated automatically on retrain. |
| `MIN_CONFIDENCE` | `0.58` | Minimum prediction confidence to open a trade. Predictions between 0.42 and 0.58 are classified as "neutral" and skipped. |
| `RETRAIN_INTERVAL` | `7200` | Seconds between automatic retraining cycles (default: 2 hours). The bot retrains using the latest downloaded data. |

---

## Data

| Variable | Default | Description |
|----------|---------|-------------|
| `CANDLE_INTERVAL` | `1h` | Candle timeframe used for both data download and prediction. |
| `DATA_DIR` | `data` | Directory where candle JSON files are stored. |
| `MAIN_LOOP_SECONDS` | `60` | Seconds between each scan iteration. Lower values scan more frequently but increase API calls. |

---

## Coin Universe

The list of coins to scan is defined directly in `config.py`:

```python
COINS: list[str] = [
    "BTC", "ETH", "SOL", "DOGE", "AVAX",
    "LINK", "ARB", "OP", "SUI", "INJ",
    "MATIC", "APT", "SEI", "TIA", "WIF",
    "PEPE", "ONDO", "RENDER", "FET", "JUP",
]
```

To add or remove coins, edit this list and re-run `download_data.py` and `train.py` to incorporate the new data.

---

## Canadian Arbitrage Scanner

The dashboard's read-only scanner checks every available spot trading-pair variant involving the configured 30 coins on NDAX, Kraken, and Coinbase, including non-CAD quote assets. It converts quote values through public spot markets to CAD when a route is available; pairs without a quote-to-CAD path are excluded. Triangular routes begin and end in CAD. Those venues serve Canadian customers; confirm your own province/account eligibility. The scanner uses public market data and does not connect with API keys or place orders.

| Variable | Default | Description |
|----------|---------|-------------|
| `CANADIAN_ARBITRAGE_EXCHANGES` | `ndax,kraken,coinbase` | Comma-separated scanner venues. Only the built-in Canadian-serving allowlist is accepted. |
| `ARBITRAGE_ASSETS` | `LTC,BCH,XLM,XRP,ADA,DOGE,SOL,TRX` | Smaller lower-transfer-cost universe used by the paper trader and whale-flow checks; the dashboard arbitrage scanner instead scans all configured coins. |
| `ARBITRAGE_NDAX_TAKER_FEE` | `0.002` | Estimated NDAX taker fee as a decimal (0.002 = 0.20%). Set your actual rate. |
| `ARBITRAGE_KRAKEN_TAKER_FEE` | `0.004` | Estimated Kraken taker fee as a decimal (0.004 = 0.40%). Set your actual rate. |
| `ARBITRAGE_COINBASE_TAKER_FEE` | `0.006` | Estimated Coinbase taker fee as a decimal (0.006 = 0.60%). Set your actual rate. |
| `ARBITRAGE_CAPITAL_CAD` | `1000` | CAD amount used to estimate opportunity profit. The scanner enforces a hard maximum of CAD $1,000 per opportunity, regardless of this setting. |
| `ARBITRAGE_FIXED_FEE_CAD_PER_ORDER` | `0` | Extra fixed charge per order, in CAD. |
| `ARBITRAGE_REBALANCE_COST_CAD` | `2` | Estimated total CAD transfer/rebalancing charge per cross-exchange round trip. |

Rates above are estimates, not live account-specific fee schedules. Taker fees are deducted from every leg, and configured fixed costs and rebalance costs are deducted from the estimated return. These costs are applied automatically and are not user-selectable in the dashboard. The scanner displays every positive-net opportunity after those costs; it has no configurable minimum-profit cutoff. The ML model controls directional paper-trade entries separately and does not invent an arbitrage threshold. Cross-exchange execution assumes you already hold inventory on both venues; transfers can take time and incur additional costs. Cross-exchange estimates walk up to 20 levels of both venue order books, match executable base-asset size, apply proportional taker fees to each fill, and stop when the next matched level is no longer profitable. The chosen CAD capital caps buy-side spend and cannot exceed $1,000 per opportunity; fixed and rebalance costs are then deducted. If either order book is unavailable, the scanner labels the result as a top-of-book estimate instead. Triangular routes remain top-of-book estimates. Neither calculation models latency, queue position, account-specific fee tiers, price movement during execution, or exchange transfer delays; all scans are read-only and are not trade instructions.

The depth-walking enhancement was informed by the general approach in [cjudice-commits/prediction-market-arb](https://github.com/cjudice-commits/prediction-market-arb). That project's Kalshi/Polymarket contracts, fee schedule, alerts, and code are not used; its market structure does not apply to Canadian crypto spot exchanges.

### Continuous ML paper trader

Run `python ml_paper_trader.py` in a separate process alongside `streamlit run dashboard.py`. It polls the smaller `ARBITRAGE_ASSETS` CAD markets every five minutes, refreshes up to 720 hourly candles hourly, checks large-trade flow every 15 minutes, and checks public headlines every 30 minutes. The ML service deliberately uses this smaller universe to limit its workload; only an explicit dashboard scan fetches the broader pair set, and order books are fetched only for potentially profitable cross-venue spreads. The service stores its simulated trade journal, learner heartbeat, model scores, and trained model in `DATA_DIR`. It does not submit orders. The dashboard's model panel reports the last heartbeat, training holdout scores, available pairs, simulated positions, and latest directional scores. The feature set includes returns and momentum, EMA/SMA trends, RSI, MACD, ATR and realized volatility, Bollinger bands, Stochastic, ADX and directional movement, CCI, Chaikin Money Flow, MFI, Donchian channels, relative volume, rolling VWAP, and OBV slope. Historical labels look four hours ahead and account for estimated round-trip fees; an 80/20 chronological holdout is reported before refitting. Closed paper trades are included in subsequent daily retraining. Recent public GDELT headlines are classified for market-moving events; new paper entries pause during recent high-impact headlines and if the catalyst service is unavailable. The dashboard also surfaces recent large public exchange prints and venue-level buy/sell imbalance. These trades do not reveal beneficial owners, wallets, or actual on-chain flows.

CUDA training is opt-in: install a CUDA-enabled PyTorch build compatible with your driver, confirm at least one GPU is visible to PyTorch, then run `python ml_paper_trader.py --device cuda`. It supports one GPU or two visible GPUs; with two, both cards are used for neural-network training. It does not use more than two GPUs, mine, or keep GPUs at 100% utilization. If no GPU is visible, the explicit CUDA option exits with an explanation. The app cannot guarantee profitability or make an exchange support short selling; shorts are paper simulations only.

RAM is detected from the operating system at startup. By default, the paper trader limits its estimated working set to 50% of currently available physical RAM and preserves the rest for Windows and other applications. Historical candle downloads are reduced to stay within that estimate; it fails clearly if there is not enough room for training data. Set `PAPER_MEMORY_FRACTION` to a value from `0.01` through `0.8` or set `PAPER_MAX_RAM_GB` to a positive hard cap. You can also pass `--max-ram-gb 4` on the command line; `0` leaves the automatic cap in effect. These are working-set limits/estimates, not operating-system-enforced process limits.

`WHALE_MIN_TRADE_CAD` (default `10000`) sets the minimum estimated CAD value of a large public exchange print; `WHALE_MEDIAN_MULTIPLE` (default `10`) requires it to also exceed that multiple of the recent median print. A sustained, strong venue-level imbalance opposing the model can block a new paper entry. These settings cannot identify a wallet or establish the true origin or destination of funds. `CATALYST_PAUSE_MINUTES` (default `60`) controls how long a recent high-impact headline can pause new entries; if the public headline service is unavailable, the paper trader conservatively pauses entries until it recovers. Neither feature affects the close logic for existing simulated positions.

---

## Example Configurations

### Conservative

Best for beginners or accounts under $1,000. Minimizes drawdown at the cost of slower growth.

```bash
LEVERAGE=3
MAX_POSITIONS=2
RISK_PER_TRADE=0.05
STOP_LOSS_PCT=0.015
TAKE_PROFIT_PCT=0.025
MAX_DAILY_LOSS_PCT=0.03
MIN_CONFIDENCE=0.62
CIRCUIT_BREAKER_LOSSES=2
CIRCUIT_BREAKER_COOLDOWN=7200
```

Key traits:

- 3x leverage limits max loss per trade
- Only 2 positions open at once
- 5% equity per trade (15% notional at 3x)
- Higher confidence threshold filters out marginal signals
- Circuit breaker trips after just 2 losses, with a 2-hour cooldown

---

### Moderate (Default)

Balanced setup suitable for accounts of $1,000-$5,000.

```bash
LEVERAGE=5
MAX_POSITIONS=3
RISK_PER_TRADE=0.10
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.03
MAX_DAILY_LOSS_PCT=0.05
MIN_CONFIDENCE=0.58
CIRCUIT_BREAKER_LOSSES=3
CIRCUIT_BREAKER_COOLDOWN=3600
```

This is the default configuration shipped with DeepAlpha. It provides a 1.5:1 reward-to-risk ratio with moderate position sizing.

---

### Aggressive

For experienced traders with accounts above $5,000 who accept higher drawdowns for faster growth.

```bash
LEVERAGE=7
MAX_POSITIONS=5
RISK_PER_TRADE=0.15
STOP_LOSS_PCT=0.025
TAKE_PROFIT_PCT=0.04
MAX_DAILY_LOSS_PCT=0.08
MIN_CONFIDENCE=0.55
CIRCUIT_BREAKER_LOSSES=4
CIRCUIT_BREAKER_COOLDOWN=1800
```

Key traits:

- 7x leverage amplifies returns (and losses)
- Up to 5 simultaneous positions
- Lower confidence threshold opens more trades
- Wider stop-loss gives trades more room to breathe
- Higher daily loss limit before stopping

!!! warning
    Aggressive settings can produce significant drawdowns. Only use these if you fully understand leverage risk and have tested on paper first.

---

## Pro Version Parameters

The [Pro version](https://stefanocrypto.gumroad.com/l/ezilv) adds these additional configuration options:

| Variable | Description |
|----------|-------------|
| `MAX_RISK_PER_TRADE_PCT` | Dynamic position sizing as % of equity (replaces fixed margin) |
| `MAX_TOTAL_EXPOSURE_PCT` | Maximum total portfolio exposure as % of equity |
| `HARD_SL_PCT` | Hard price stop-loss -- absolute max loss per position (e.g., 1% price = 5% at 5x) |
| `TRAIL_TIER1_PCT` | Trailing TP tier 1: move SL to breakeven when profit exceeds this level |
| `TRAIL_TIER2_PCT` | Trailing TP tier 2: lock partial profit when profit exceeds this level |
| `COIN_BLACKLIST` | Comma-separated list of coins to exclude from scanning |
| `GLOBAL_OPEN_COOLDOWN` | Minimum seconds between opening any two positions |
