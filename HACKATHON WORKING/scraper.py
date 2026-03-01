import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Settings ──────────────────────────────────────────────────────────────

OUTPUT_FILE    = "reddit_sentiment.parquet"
POSTS_PER_QUERY = 100
SORT_BY        = "new"
TIME_FILTER    = "month"
REQUEST_DELAY  = 0.5

SUBREDDITS = ["stocks", "investing", "wallstreetbets", "SecurityAnalysis", "StockMarket"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
})

# ── Reddit fetching ───────────────────────────────────────────────────────

def _fetch_subscriber_count(subreddit: str) -> tuple:
    try:
        r = SESSION.get(f"https://www.reddit.com/r/{subreddit}/about.json", timeout=10)
        if r.status_code == 200:
            return subreddit, r.json()["data"].get("subscribers", 0)
    except Exception as e:
        log.warning(f"Subscriber count failed for r/{subreddit}: {e}")
    return subreddit, 0


def _fetch_posts(keyword: str, subreddit: str) -> list:
    url = (f"https://www.reddit.com/r/{subreddit}/search.json"
           f"?q={keyword}&restrict_sr=1&sort={SORT_BY}&t={TIME_FILTER}&limit={POSTS_PER_QUERY}")
    try:
        r = SESSION.get(url, timeout=10)
        if r.status_code == 429:
            log.warning("Rate limited — sleeping 30s")
            time.sleep(30)
            return []
        if r.status_code != 200:
            log.error(f"r/{subreddit} '{keyword}' — HTTP {r.status_code}")
            return []
        posts = r.json()["data"]["children"]
        log.info(f"r/{subreddit} '{keyword}': {len(posts)} posts")
        return posts
    except Exception as e:
        log.error(f"Error fetching r/{subreddit} '{keyword}': {e}")
        return []


def _parse_posts(posts: list, stock_name: str, subreddit: str, subscribers: int) -> list:
    out = []
    for post in posts:
        try:
            d = post["data"]
            if d.get("stickied"):
                continue
            if d.get("title") in [None, "[deleted]", "[removed]"]:
                continue
            out.append({
                "title":      d.get("title", "").strip(),
                "source":     "reddit",
                "date":       datetime.utcfromtimestamp(d.get("created_utc", 0)).strftime("%Y-%m-%d"),
                "subscribers": subscribers,
                "post_url":   f"https://reddit.com{d.get('permalink', '')}",
            })
        except Exception as e:
            log.warning(f"Skipped post: {e}")
    return out


def run_scraper(ticker: str, name: str):
    """Fetch Reddit posts concurrently and save to OUTPUT_FILE."""
    # Subscriber counts in parallel
    subscriber_counts = {}
    with ThreadPoolExecutor(max_workers=len(SUBREDDITS)) as ex:
        for sub, count in ex.map(_fetch_subscriber_count, SUBREDDITS):
            subscriber_counts[sub] = count
            log.info(f"r/{sub}: {count:,} subscribers")

    # Posts in parallel
    all_rows = []
    tasks = [(kw, sub) for kw in [ticker, name] for sub in SUBREDDITS]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_posts, kw, sub): (kw, sub) for kw, sub in tasks}
        for fut in as_completed(futures):
            kw, sub = futures[fut]
            posts = fut.result()
            all_rows.extend(_parse_posts(posts, name, sub, subscriber_counts[sub]))
            time.sleep(REQUEST_DELAY)

    if not all_rows:
        log.error("No Reddit posts found.")
        return

    df = pd.DataFrame(all_rows)
    df.drop_duplicates(subset=["post_url"], inplace=True)
    df = df[["title", "source", "date", "subscribers"]].sort_values("date", ascending=False)
    df.to_parquet(OUTPUT_FILE, index=False)
    log.info(f"Saved {len(df)} posts to {OUTPUT_FILE}")

# ── Sentiment — scored once, cached ──────────────────────────────────────

_reddit_cache: dict = {}

def calculate_reddit_sentiment(file_path: str, end_date_str: str, num_days: int) -> list:
    """Score Reddit parquet once with reach-weighting, cache, aggregate per window."""
    from sentiment import load_and_score, aggregate_sentiment
    if file_path not in _reddit_cache:
        if not os.path.exists(file_path):
            return [0.0, 1.0]
        _reddit_cache[file_path] = load_and_score(file_path, subscribers_col=True)
    return aggregate_sentiment(_reddit_cache[file_path], end_date_str, num_days)


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Reddit Scraper ────────────────────────")
    ticker   = input("  Ticker : ").strip().upper()
    name     = input("  Name   : ").strip()
    lookback = int(input("  Lookback (days): ").strip())
    run_date = datetime.now().strftime("%Y-%m-%d")
    print(f"  Date   : {run_date}\n")

    run_scraper(ticker, name)
    s = calculate_reddit_sentiment(OUTPUT_FILE, run_date, lookback)
    val = s[0] / s[1] if s[1] != 0 else 0.0
    label = "POSITIVE 🟢" if val > 0.05 else ("NEGATIVE 🔴" if val < -0.05 else "NEUTRAL ⚪")
    print(f"\n  Reddit Sentiment → {val:+.4f}  ({label})\n")
