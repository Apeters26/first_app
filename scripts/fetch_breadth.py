"""Build data/breadth.json: a handful of S&P 500 market-breadth indicators.

Free sources only:
  - S&P 500 member list: Wikipedia (fallback: datasets/s-and-p-500-companies on GitHub)
  - Daily prices: Yahoo Finance via the yfinance package

Run:  python scripts/fetch_breadth.py            (live data)
      python scripts/fetch_breadth.py --demo     (synthetic data, for offline testing)
"""

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "breadth.json"
UA = {"User-Agent": "Mozilla/5.0 (market-breadth dashboard)"}
KEEP_DAYS = 260  # about one trading year shown in the app


def sp500_tickers():
    try:
        html = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30
        ).text
        syms = pd.read_html(io.StringIO(html))[0]["Symbol"]
    except Exception as exc:  # noqa: BLE001
        print(f"Wikipedia failed ({exc}); using GitHub dataset", file=sys.stderr)
        csv = requests.get(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
            headers=UA,
            timeout=30,
        ).text
        syms = pd.read_csv(io.StringIO(csv))["Symbol"]
    return sorted({s.strip().replace(".", "-") for s in syms})


def download_closes(tickers):
    import yfinance as yf

    data = yf.download(
        tickers, period="2y", interval="1d", auto_adjust=True, progress=False, threads=True
    )
    closes = data["Close"].dropna(how="all")
    # Drop the last row if most stocks have no price yet (partial day).
    if closes.iloc[-1].notna().mean() < 0.8:
        closes = closes.iloc[:-1]
    return closes


def demo_closes():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)
    market = rng.normal(0.0004, 0.01, len(dates))
    cols = {}
    for i in range(500):
        beta = rng.uniform(0.6, 1.4)
        rets = beta * market + rng.normal(0, 0.015, len(dates))
        cols[f"S{i:03d}"] = 100 * np.exp(np.cumsum(rets))
    cols["SPY"] = 400 * np.exp(np.cumsum(market * 1.1))
    cols["RSP"] = 150 * np.exp(np.cumsum(market))
    return pd.DataFrame(cols, index=dates)


def status_from(value, good, bad):
    """good/bad are thresholds; higher is better."""
    if value >= good:
        return "healthy"
    if value <= bad:
        return "weak"
    return "mixed"


def series(s, decimals=1):
    s = s.iloc[-KEEP_DAYS:]
    return [None if pd.isna(v) else round(float(v), decimals) for v in s]


def build(closes):
    spy, rsp = closes["SPY"], closes["RSP"]
    stocks = closes.drop(columns=["SPY", "RSP"])
    dates = [d.strftime("%Y-%m-%d") for d in closes.index[-KEEP_DAYS:]]

    def pct_above(window):
        ma = stocks.rolling(window, min_periods=window).mean()
        valid = ma.notna() & stocks.notna()
        above = ((stocks > ma) & valid).sum(axis=1)
        return above / valid.sum(axis=1).replace(0, np.nan) * 100

    p200, p50 = pct_above(200), pct_above(50)

    hi = stocks.rolling(252, min_periods=200).max()
    lo = stocks.rolling(252, min_periods=200).min()
    net_hl = (stocks >= hi).sum(axis=1) - (stocks <= lo).sum(axis=1)
    net_hl_avg = net_hl.rolling(10).mean()

    chg = stocks.diff()
    adv, dec = (chg > 0).sum(axis=1), (chg < 0).sum(axis=1)
    ad_line = (adv - dec).iloc[1:].cumsum()
    rana = (adv - dec) / (adv + dec).replace(0, np.nan) * 1000
    mcclellan = rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()
    ad_ma = ad_line.rolling(50).mean()

    ratio = rsp / spy
    ratio_idx = ratio / ratio.iloc[-KEEP_DAYS] * 100
    ratio_3m = (ratio.iloc[-1] / ratio.iloc[-64] - 1) * 100

    v200, v50 = p200.iloc[-1], p50.iloc[-1]
    vhl, vmc = net_hl_avg.iloc[-1], mcclellan.iloc[-1]
    ad_up = ad_line.iloc[-1] > ad_ma.iloc[-1]

    s_ad = "healthy" if ad_up and vmc > 0 else "weak" if not ad_up and vmc < 0 else "mixed"
    indicators = [
        {
            "id": "p200",
            "title": "Stocks above 200-day average",
            "value": round(v200, 1),
            "unit": "%",
            "status": status_from(v200, 60, 40),
            "what": "Share of S&P 500 stocks in a long-term uptrend. Above 60% is broad strength; below 40% means most stocks are struggling.",
            "series": series(p200),
            "ref": 50,
        },
        {
            "id": "p50",
            "title": "Stocks above 50-day average",
            "value": round(v50, 1),
            "unit": "%",
            "status": status_from(v50, 60, 40),
            "what": "Same idea over a shorter window, so it reacts faster. Readings above 80% or below 20% are often stretched.",
            "series": series(p50),
            "ref": 50,
        },
        {
            "id": "hl",
            "title": "New highs minus new lows",
            "value": round(vhl, 1),
            "unit": "",
            "status": status_from(vhl, 5, -5),
            "what": "Stocks at a 52-week high minus those at a 52-week low (10-day average). Positive means more leaders than laggards.",
            "series": series(net_hl_avg),
            "ref": 0,
        },
        {
            "id": "ad",
            "title": "Advance / decline line",
            "value": round(vmc, 1),
            "unit": "",
            "value_label": "McClellan oscillator",
            "status": s_ad,
            "what": "Running total of rising minus falling stocks each day. A rising line means most stocks are joining in; the McClellan oscillator shows its short-term momentum (above 0 is good).",
            "series": series(ad_line, 0),
            "ref": None,
        },
        {
            "id": "ew",
            "title": "Equal-weight vs. S&P 500",
            "value": round(ratio_3m, 1),
            "unit": "%",
            "value_label": "3-month change (RSP / SPY)",
            "status": status_from(ratio_3m, 1, -1),
            "what": "Compares the average stock (equal-weight RSP) to the cap-weighted index (SPY). Falling means a few giant companies are carrying the market.",
            "series": series(ratio_idx, 2),
            "ref": 100,
        },
    ]

    score = sum({"healthy": 1, "weak": -1}.get(i["status"], 0) for i in indicators)
    overall = "healthy" if score >= 2 else "weak" if score <= -2 else "mixed"
    healthy = sum(i["status"] == "healthy" for i in indicators)
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "asof": dates[-1],
        "universe": int(stocks.iloc[-1].notna().sum()),
        "overall": overall,
        "healthy_count": healthy,
        "total": len(indicators),
        "spy_close": round(float(spy.iloc[-1]), 2),
        "spy_3m": round(float((spy.iloc[-1] / spy.iloc[-64] - 1) * 100), 1),
        "dates": dates,
        "indicators": indicators,
    }


def main():
    if "--demo" in sys.argv:
        closes = demo_closes()
    else:
        tickers = sp500_tickers()
        print(f"{len(tickers)} tickers", file=sys.stderr)
        closes = download_closes(tickers + ["SPY", "RSP"])
        if closes.shape[1] < 400:
            sys.exit(f"Only got {closes.shape[1]} tickers; refusing to overwrite data")
    data = build(closes)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {OUT} (as of {data['asof']}, overall {data['overall']})", file=sys.stderr)


if __name__ == "__main__":
    main()
