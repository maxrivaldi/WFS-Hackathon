import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ── Go scraper ────────────────────────────────────────────────────────────

def run_go_scraper(file_name: str, target_date: str, days: int, ticker: str, stock: str):
    import os
    abs_file = os.path.abspath(file_name)
    cmd = ["go", "run", abs_file,
           f"-start={target_date}", f"-days={days}",
           f"-ticker={ticker}", f"-stock={stock}"]
    print(f"  Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(abs_file))
    if result.returncode != 0:
        raise RuntimeError(f"Go scraper failed (exit {result.returncode})")

# ── Shared FinBERT — loaded once, reused everywhere ───────────────────────

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        print("  Loading FinBERT model...")
        name = "ProsusAI/finbert"
        _nlp = pipeline(
            "sentiment-analysis",
            model=AutoModelForSequenceClassification.from_pretrained(name),
            tokenizer=AutoTokenizer.from_pretrained(name),
            device=-1,
        )
    return _nlp

LABEL_SIGN = {"negative": -1, "neutral": 0, "positive": 1}

# ── Core scoring functions ────────────────────────────────────────────────

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run FinBERT on df['title'] once. Returns df with 'finbert_score' added."""
    results = get_nlp()(df["title"].tolist())
    df = df.copy()
    df["finbert_score"] = [
        (2 if r["label"] == "negative" else 1) * r["score"] * LABEL_SIGN[r["label"]]
        for r in results
    ]
    return df


def load_and_score(file_path: str, subscribers_col: bool = False) -> pd.DataFrame:
    """Load parquet, drop blanks, run FinBERT once, return scored DataFrame."""
    cols = ["title", "date"] + (["subscribers"] if subscribers_col else [])
    available = pd.read_parquet(file_path).columns.tolist()
    cols = [c for c in cols if c in available]
    df = pd.read_parquet(file_path, columns=cols)
    df["date"]  = df["date"].astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df = df[df["title"].str.strip() != ""].copy()
    if df.empty:
        return df
    df = score_dataframe(df)
    if "subscribers" in df.columns:
        raw_reach = (df["subscribers"] ** 0.25) * np.log1p(df["subscribers"])
        max_reach = raw_reach.max()
        if max_reach > 0:
            df["finbert_score"] *= (raw_reach / max_reach).values
    return df


def aggregate_sentiment(scored_df: pd.DataFrame, end_date_str: str, num_days: int) -> list:
    """
    Given a pre-scored DataFrame, return [numerator, denominator] for the
    date window [end_date - num_days, end_date].
    """
    if scored_df.empty:
        return [0.0, 1.0]

    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    date_set = {
        (end_date - timedelta(days=x)).strftime("%Y-%m-%d")
        for x in range(num_days)
    }
    df = scored_df[scored_df["date"].isin(date_set)]
    if df.empty:
        return [0.0, 1.0]

    daily = df.groupby("date").agg(
        count=("finbert_score", "count"),
        avg_score=("finbert_score", "mean"),
    ).sort_index()

    daily["hype"] = np.exp(daily["count"] - daily["count"].max())
    dates_dt      = pd.to_datetime(daily.index)
    t_days        = (dates_dt.max() - dates_dt).days
    daily["recency"] = 1 - np.tanh(t_days / 3)

    w = daily["hype"] * daily["recency"]
    total = w.sum()
    if total == 0:
        return [0.0, 1.0]
    return [(daily["avg_score"] * w).sum(), total]


# ── Cached public API ─────────────────────────────────────────────────────

_news_cache: dict = {}

def calculate_weighted_sentiment(file_path: str, end_date_str: str, num_days: int) -> list:
    """Score news parquet once, cache, then aggregate per window call."""
    if file_path not in _news_cache:
        _news_cache[file_path] = load_and_score(file_path, subscribers_col=False)
    return aggregate_sentiment(_news_cache[file_path], end_date_str, num_days)
