import os
import sys
import json
import base64
import pathlib
import threading
import traceback
import warnings
import logging

# Must be set before any matplotlib import anywhere in the process
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no GUI, no multiprocessing

warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")

from flask import Flask, request, jsonify, send_from_directory

_HERE = pathlib.Path(__file__).parent.resolve()
app = Flask(__name__, static_folder=str(_HERE / "static"))

# Suppress Flask request logs from cluttering the terminal
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# ── Shared state ──────────────────────────────────────────────────────────

_state = {
    "running": False,
    "done":    False,
    "logs":    [],
    "result":  None,
}

def _reset():
    _state["running"] = False
    _state["done"]    = False
    _state["logs"]    = []
    _state["result"]  = None

def emit(msg: str, kind: str = "log"):
    _state["logs"].append({"kind": kind, "msg": str(msg)})

# ── Log handler that routes Python logging → UI ───────────────────────────
# We do NOT monkey-patch builtins.print — that causes issues with
# multiprocessing (matplotlib uses it internally). Instead we install a
# logging.Handler so that any module using the standard logger is captured.

class _UIHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        kind = "error" if record.levelno >= logging.ERROR else \
               "success" if record.levelno == logging.DEBUG else "log"
        emit(msg, kind)

_ui_handler = _UIHandler()
_ui_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(_ui_handler)
logging.getLogger().setLevel(logging.INFO)

# ── Analysis thread ───────────────────────────────────────────────────────

def run_analysis(ticker, name, lookback, end_date):
    try:
        # These imports happen inside the thread so _HERE is already set
        sys.path.insert(0, str(_HERE))
        from sentiment import run_go_scraper
        from scraper import run_scraper, OUTPUT_FILE
        import tsallis as T

        emit(f"▶  {ticker} ({name})  |  {lookback}d  |  {end_date}", "header")
        emit("", "spacer")

        emit("── [1/4] Fetching Finnhub news ──────────────", "section")
        run_go_scraper("yahooscrape.go", end_date, lookback, ticker, name)
        emit("✓  Done", "success")
        emit("", "spacer")

        emit("── [2/4] Scraping Reddit posts ──────────────", "section")
        run_scraper(ticker, name)
        emit("✓  Done", "success")
        emit("", "spacer")

        emit("── [3/4] Running analysis ───────────────────", "section")
        res = T.get_score(
            ticker=ticker,
            news_parquet="mentions.parquet",
            reddit_parquet=OUTPUT_FILE,
            end_date=end_date,
            lookback=lookback,
        )
        emit("✓  Done", "success")
        emit("", "spacer")

        emit("── [4/4] Generating chart ───────────────────", "section")
        T.plot_signal_vs_price(
            ticker=ticker,
            news_parquet="mentions.parquet",
            reddit_parquet=OUTPUT_FILE,
            end_date=end_date,
            lookback=lookback,
        )
        emit("✓  Chart ready", "success")

        # Read chart as base64
        chart_path = str(_HERE / f"{ticker}_signal_chart.png")
        chart_b64 = ""
        if os.path.exists(chart_path):
            with open(chart_path, "rb") as f:
                chart_b64 = base64.b64encode(f.read()).decode()

        if res.score < 0.3:
            verdict, verdict_cls = "SIT OUT", "sit"
        elif res.score > 0.6:
            verdict     = "BUY" if res.trade_signal > 0 else "SELL"
            verdict_cls = "buy" if res.trade_signal > 0 else "sell"
        else:
            verdict, verdict_cls = "CAUTION", "caution"

        _state["result"] = {
            "ticker":       ticker,
            "date":         end_date,
            "regime":       res.regime,
            "vol_score":    round(res.score, 4),
            "news_sent":    round(res.news_sent, 4),
            "reddit_sent":  round(res.reddit_sent, 4),
            "combined":     round(res.final_sent, 4),
            "trade_signal": round(res.trade_signal, 4),
            "verdict":      verdict,
            "verdict_cls":  verdict_cls,
            "chart_b64":    chart_b64,
        }

    except Exception as e:
        emit(f"✗  {e}", "error")
        emit(traceback.format_exc(), "error")
    finally:
        _state["running"] = False
        _state["done"]    = True

# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(_HERE / "static"), "index.html")

@app.route("/run", methods=["POST"])
def run():
    if _state["running"]:
        return jsonify({"error": "Already running"}), 409
    data     = request.json
    ticker   = data.get("ticker", "").strip().upper()
    name     = data.get("name", "").strip()
    lookback = int(data.get("lookback", 30))
    from datetime import datetime
    end_date = data.get("end_date", "").strip() or datetime.now().strftime("%Y-%m-%d")

    _reset()
    _state["running"] = True
    threading.Thread(
        target=run_analysis,
        args=(ticker, name, lookback, end_date),
        daemon=True
    ).start()
    return jsonify({"status": "started"})

@app.route("/logs")
def logs():
    return jsonify({
        "running": _state["running"],
        "done":    _state["done"],
        "logs":    _state["logs"],
        "result":  _state["result"],
    })

def _warmup():
    """Pre-load all heavy modules in the background so the first Run is fast."""
    try:
        sys.path.insert(0, str(_HERE))
        import numpy, pandas, yfinance, matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot
        import sentiment, scraper, tsallis
        emit("  ✓  Models ready", "success")
    except Exception as e:
        emit(f"  Warmup warning: {e}", "log")

if __name__ == "__main__":
    os.chdir(str(_HERE))
    # Load heavy dependencies in background — server starts instantly,
    # imports finish before the user has filled in the form
    threading.Thread(target=_warmup, daemon=True).start()
    print(f"  Starting server → http://localhost:5001")
    app.run(debug=False, host="0.0.0.0", port=5001, threaded=True)
