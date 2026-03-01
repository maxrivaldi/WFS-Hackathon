import matplotlib
matplotlib.use('Agg')  # non-interactive — safe for server use

import os
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────

Q             = 0.5
LAMBDA_DECAY  = 0.97
WINDOW        = 60
N_BINS        = 15
BIN_SCALE     = 3.5
MULTI_WINDOWS = (30, 60, 120)
PARK_WEIGHT   = 0.3
MIN_BARS      = 10

# ── Volatility helpers ────────────────────────────────────────────────────

def _parkinson(highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    safe_lows = np.where(lows > 0, lows, highs)
    log_hl_sq = np.log(np.where(highs > safe_lows, highs / safe_lows, 1.0)) ** 2
    out = np.full(len(highs), np.nan)
    for i in range(WINDOW, len(highs)):
        out[i] = np.sqrt(np.mean(log_hl_sq[i - WINDOW:i]) / (4.0 * np.log(2.0)))
    return out

def _norm(value: float, past: np.ndarray) -> float:
    clean = past[~np.isnan(past)]
    if len(clean) < 5 or np.std(clean) == 0:
        return 0.5
    z = (value - np.mean(clean)) / np.std(clean)
    return float(np.clip(np.tanh(z / 2.0 + 0.5), 0.0, 1.0))

def _weights(T: int) -> np.ndarray:
    return LAMBDA_DECAY ** np.arange(T, 0, -1, dtype=np.float64)

def _weighted_probs(returns: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bins = len(edges) - 1
    wp = np.zeros(n_bins)
    idx = np.clip(np.digitize(returns, edges) - 1, 0, n_bins - 1)
    np.add.at(wp, idx, weights)
    s = weights.sum()
    return wp / s if s > 0 else wp

def _entropy(wp: np.ndarray) -> float:
    nz = wp[wp > 0]
    return (1.0 - np.sum(nz ** Q)) / (Q - 1.0) if len(nz) > 0 else 0.0

def _tsallis_score(returns: np.ndarray, edges: np.ndarray, window: int) -> float:
    eff = returns[-window:] if len(returns) >= window else returns
    wp  = _weighted_probs(eff, _weights(len(eff)), edges)
    mx  = (N_BINS ** (1.0 - Q) - 1.0) / (1.0 - Q)
    return float(np.clip(_entropy(wp) / mx, 0.0, 1.0)) if mx else 0.0

def _vol_score(returns: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> float:
    r = returns[:-1].astype(np.float64)
    if len(r) < 2 or np.std(r) == 0:
        tsallis = 0.5
    else:
        half  = BIN_SCALE * np.std(r)
        edges = np.linspace(np.mean(r) - half, np.mean(r) + half, N_BINS + 1)
        valid = [w for w in MULTI_WINDOWS if len(returns) >= w]
        scores = [_tsallis_score(returns, edges, w) for w in valid]
        tsallis = float(np.mean(scores)) if scores else 0.5

    park = 0.5
    if PARK_WEIGHT > 0 and highs is not None and len(highs) > WINDOW:
        pv = _parkinson(highs, lows)
        if not np.isnan(pv[-1]):
            park = _norm(pv[-1], pv[:-1])

    return float(np.clip((1.0 - PARK_WEIGHT) * tsallis + PARK_WEIGHT * park, 0.0, 1.0))

# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class Result:
    score:        float   # volatility [0, 1]
    news_sent:    float   # news sentiment [-1, 1]
    reddit_sent:  float   # reddit sentiment [-1, 1]
    final_sent:   float   # combined sentiment [-1, 1]
    trade_signal: float   # final_sent * score
    regime:       str     # LOW / MODERATE / HIGH

# ── Main analysis function ────────────────────────────────────────────────

def get_score(ticker: str, news_parquet: str, reddit_parquet: str,
              end_date: str, lookback: int) -> Result:
    from sentiment import calculate_weighted_sentiment
    from scraper import calculate_reddit_sentiment

    end_dt   = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=lookback)

    # Price data (daily bars — always available for any date range)
    df = yf.Ticker(ticker).history(
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
    )
    if df.empty:
        raise ValueError(f"No price data returned for '{ticker}'. Check the ticker symbol.")

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna(subset=["log_return"])

    if len(df) < MIN_BARS:
        raise ValueError(f"Only {len(df)} trading days in range — try a longer lookback.")

    returns = df["log_return"].values
    highs   = df["High"].values
    lows    = df["Low"].values

    vol = _vol_score(returns, highs, lows)

    # Sentiment — FinBERT runs once per file, subsequent calls use cache
    news   = calculate_weighted_sentiment(news_parquet,   end_date, lookback)
    reddit = calculate_reddit_sentiment(reddit_parquet,   end_date, lookback)

    news_val   = news[0]   / news[1]   if news[1]   != 0 else 0.0
    reddit_val = reddit[0] / reddit[1] if reddit[1] != 0 else 0.0
    combined   = (news[0] + reddit[0]) / (news[1] + reddit[1]) if (news[1] + reddit[1]) != 0 else 0.0

    signal = combined * vol
    regime = "LOW" if vol < 0.3 else "MODERATE" if vol < 0.6 else "HIGH"

    return Result(score=vol, news_sent=news_val, reddit_sent=reddit_val,
                  final_sent=combined, trade_signal=signal, regime=regime)

# ── Display ───────────────────────────────────────────────────────────────

def _bar(v: float, w: int = 26) -> str:
    f = int(round(max(0.0, min(1.0, v)) * w))
    return f"[{'█'*f}{'░'*(w-f)}] {v:.2f}"

def _sbar(v: float, w: int = 26) -> str:
    c = w // 2
    m = min(int(round(abs(v) * c)), c)
    bar = ("─"*c + "█"*m + "░"*(c-m)) if v >= 0 else ("░"*(c-m) + "█"*m + "─"*c)
    return f"[{bar}] {v:+.2f}"

def print_result(ticker: str, res: Result, date: str):
    W   = 52
    div = "─" * W
    if res.score < 0.3:
        verdict = "⚪  SIT OUT          (low market activity)"
    elif res.score > 0.6:
        verdict = f"FULL CONVICTION  →  {'BUY 🟢' if res.trade_signal > 0 else 'SELL 🔴'}"
    else:
        verdict = "🟡  MONITOR"

    print()
    print(f"┌{div}┐")
    print(f"│{'  '+ticker+'  ·  '+date:^{W}}│")
    print(f"├{div}┤")
    print(f"│  {'Volatility Regime':<20} {res.regime+' VOLATILITY':>{W-22}}  │")
    print(f"├{div}┤")
    print(f"│  Vol Score    {_bar(res.score)}  │")
    print(f"│  News Sent.   {_sbar(res.news_sent)}  │")
    print(f"│  Reddit Sent. {_sbar(res.reddit_sent)}  │")
    print(f"│  Combined     {_sbar(res.final_sent)}  │")
    print(f"│  Trade Signal {_sbar(res.trade_signal)}  │")
    print(f"├{div}┤")
    print(f"│  Verdict:  {verdict:<{W-12}}│")
    print(f"└{div}┘")
    print()

# ── Chart ─────────────────────────────────────────────────────────────────

def plot_signal_vs_price(ticker: str, news_parquet: str, reddit_parquet: str,
                         end_date: str, lookback: int):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from sentiment import load_and_score, aggregate_sentiment

    end_dt   = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=lookback)

    # Daily price for the window
    price_df = yf.Ticker(ticker).history(
        start=start_dt.strftime("%Y-%m-%d"),
        end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d", auto_adjust=True,
    )
    if price_df.empty:
        print("  ✗  No price data for chart.")
        return

    price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
    price_df["pct"] = price_df["Close"].diff()
    trading_days    = price_df.index[1:]

    # Score ALL headlines once upfront
    print("  Scoring news headlines (one-time)...")
    news_scored = load_and_score(news_parquet) if os.path.exists(news_parquet) else pd.DataFrame()

    print("  Scoring Reddit headlines (one-time)...")
    reddit_scored = pd.DataFrame()
    if os.path.exists(reddit_parquet):
        reddit_scored = load_and_score(reddit_parquet, subscribers_col=True)

    # Slide window — only cheap aggregation inside the loop
    print(f"  Computing signal for {len(trading_days)} trading days...")
    dates, signals = [], []

    for dt in trading_days:
        dt_str = dt.strftime("%Y-%m-%d")

        n = aggregate_sentiment(news_scored,   dt_str, lookback) if not news_scored.empty   else [0.0, 1.0]
        r = aggregate_sentiment(reddit_scored, dt_str, lookback) if not reddit_scored.empty else [0.0, 1.0]
        den = n[1] + r[1]
        sentiment = (n[0] + r[0]) / den if den != 0 else 0.0

        hist     = price_df[price_df.index <= dt]
        log_rets = np.log(hist["Close"] / hist["Close"].shift(1)).dropna().values
        if len(log_rets) < 2:
            signals.append(0.0)
            dates.append(dt)
            continue

        rv = log_rets[:-1].astype(np.float64)
        if len(rv) < 2 or np.std(rv) == 0:
            vol = 0.5
        else:
            half  = BIN_SCALE * np.std(rv)
            edges = np.linspace(np.mean(rv) - half, np.mean(rv) + half, N_BINS + 1)
            valid = [w for w in MULTI_WINDOWS if len(log_rets) >= w]
            vol   = float(np.mean([_tsallis_score(log_rets, edges, w) for w in valid])) if valid else 0.5

        signals.append(sentiment * vol)
        dates.append(dt)

    chart_df = pd.DataFrame({
        "date":   dates,
        "signal": signals,
        "pct":    price_df.loc[trading_days, "pct"].values,
    }).dropna()

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f0f")
    ax1.set_facecolor("#0f0f0f")

    colors = ["#26a65b" if v >= 0 else "#e84545" for v in chart_df["pct"]]
    ax1.bar(chart_df["date"], chart_df["pct"], color=colors, alpha=0.45, width=0.8, label="Daily Change")
    ax1.axhline(0, color="#444", linewidth=0.8)
    ax1.set_ylabel("Daily Price Change", color="#aaaaaa", fontsize=10)
    ax1.tick_params(axis="y", colors="#aaaaaa")
    ax1.tick_params(axis="x", colors="#aaaaaa", rotation=35)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    ax2 = ax1.twinx()
    ax2.plot(chart_df["date"], chart_df["signal"], color="#f0c040", linewidth=2.2,
             label="Trade Signal", zorder=5)
    ax2.axhline(0, color="#555", linewidth=0.6, linestyle="--")
    ax2.set_ylabel("Trade Signal", color="#f0c040", fontsize=10)
    ax2.tick_params(axis="y", colors="#f0c040")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper left",
               facecolor="#1a1a1a", edgecolor="#333", labelcolor="white", fontsize=9)

    plt.title(f"{ticker}  —  Trade Signal vs Daily Price Change",
              color="white", fontsize=13, pad=12)
    fig.tight_layout()

    out = f"{ticker}_signal_chart.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"  Chart saved → {out}")

# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sentiment import run_go_scraper
    from scraper import run_scraper, OUTPUT_FILE

    print("\n── Stock Analysis Setup ──────────────────")
    ticker   = input("  Ticker symbol  : ").strip().upper()
    name     = input("  Company name   : ").strip()
    lookback = int(input("  Lookback (days): ").strip())
    raw_date = input("  End date (YYYY-MM-DD, Enter = today): ").strip()
    if raw_date:
        try:
            datetime.strptime(raw_date, "%Y-%m-%d")
            end_date = raw_date
        except ValueError:
            print(f"  Invalid date '{raw_date}', using today.")
            end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")
    print(f"  Analysis date  : {end_date}")
    print("──────────────────────────────────────────\n")

    print("  [1/4] Fetching Finnhub news...")
    run_go_scraper("yahooscrape.go", end_date, lookback, ticker, name)

    print("  [2/4] Scraping Reddit posts...")
    run_scraper(ticker, name)

    print("  [3/4] Running analysis...\n")
    try:
        res = get_score(
            ticker=ticker,
            news_parquet="mentions.parquet",
            reddit_parquet=OUTPUT_FILE,
            end_date=end_date,
            lookback=lookback,
        )
        print_result(ticker, res, end_date)

        print("  [4/4] Generating chart...")
        plot_signal_vs_price(
            ticker=ticker,
            news_parquet="mentions.parquet",
            reddit_parquet=OUTPUT_FILE,
            end_date=end_date,
            lookback=lookback,
        )
    except Exception as e:
        print(f"\n  ✗  Error: {e}\n")