"""Build data/breadth.json: market-breadth indicators, a VIX gauge and a Stockbee-style monitor.

Free sources only:
  - S&P 500 / 400 / 600 member lists: Wikipedia (S&P 500 fallback: datasets/s-and-p-500-companies)
  - Daily prices and volume: Yahoo Finance via the yfinance package

The five breadth checks use the S&P 500. The Stockbee-style monitor uses the
wider S&P 1500 (500 + 400 + 600) so the big-mover counts are meaningful.

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
MONITOR_DAYS = 10  # rows in the Stockbee-style table
EXTRAS = ["SPY", "RSP", "^VIX", "^VIX3M"]
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_{}_companies"


def wiki_symbols(n):
    html = requests.get(WIKI.format(n), headers=UA, timeout=30).text
    for table in pd.read_html(io.StringIO(html)):
        for col in table.columns:
            if str(col).lower() in ("symbol", "ticker symbol", "ticker"):
                return {str(s).strip().replace(".", "-") for s in table[col].dropna()}
    raise ValueError(f"no symbol column on S&P {n} page")


def load_universe():
    try:
        sp500 = wiki_symbols(500)
    except Exception as exc:  # noqa: BLE001
        print(f"Wikipedia S&P 500 failed ({exc}); using GitHub dataset", file=sys.stderr)
        csv = requests.get(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
            headers=UA,
            timeout=30,
        ).text
        sp500 = {s.strip().replace(".", "-") for s in pd.read_csv(io.StringIO(csv))["Symbol"]}
    wider = set(sp500)
    for n in (400, 600):
        try:
            wider |= wiki_symbols(n)
        except Exception as exc:  # noqa: BLE001
            print(f"S&P {n} list failed ({exc}); monitor uses a smaller universe", file=sys.stderr)
    return sorted(sp500), sorted(wider)


def download(tickers):
    import yfinance as yf

    data = yf.download(
        tickers, period="2y", interval="1d", auto_adjust=True, progress=False, threads=True
    )
    close, volume = data["Close"].dropna(how="all"), data["Volume"]
    # Drop the last row if most stocks have no price yet (partial day).
    if close.iloc[-1].notna().mean() < 0.8:
        close = close.iloc[:-1]
    return close, volume.reindex(close.index)


def demo_data():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)
    market = rng.normal(0.0004, 0.01, len(dates))
    close, vol = {}, {}
    for i in range(1500):
        beta = rng.uniform(0.6, 1.6)
        rets = beta * market + rng.normal(0, 0.02 if i >= 500 else 0.015, len(dates))
        close[f"S{i:04d}"] = 50 * np.exp(np.cumsum(rets))
        vol[f"S{i:04d}"] = rng.lognormal(13.5, 0.5, len(dates))
    close["SPY"] = 400 * np.exp(np.cumsum(market * 1.1))
    close["RSP"] = 150 * np.exp(np.cumsum(market))
    vix = np.clip(18 - 400 * pd.Series(market).rolling(10, min_periods=1).mean() + rng.normal(0, 1, len(dates)), 10, 60)
    close["^VIX"] = vix.values
    close["^VIX3M"] = (vix * 0.6 + 8).values
    tickers = [f"S{i:04d}" for i in range(1500)]
    return pd.DataFrame(close, index=dates), pd.DataFrame(vol, index=dates), tickers[:500], tickers


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


def pct_above(prices, window):
    ma = prices.rolling(window, min_periods=window).mean()
    valid = ma.notna() & prices.notna()
    above = ((prices > ma) & valid).sum(axis=1)
    return above / valid.sum(axis=1).replace(0, np.nan) * 100


def breadth_checks(close, sp500):
    spy, rsp = close["SPY"], close["RSP"]
    stocks = close[[t for t in sp500 if t in close]]
    p200, p50 = pct_above(stocks, 200), pct_above(stocks, 50)

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

    return int(stocks.iloc[-1].notna().sum()), [
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


def vix_card(close):
    vix, vix3m = close["^VIX"].ffill(), close["^VIX3M"].ffill()
    v, term = float(vix.iloc[-1]), float(vix.iloc[-1] / vix3m.iloc[-1])
    if v >= 30 or term >= 1:
        status, label = "weak", "Fear"
    elif v >= 20:
        status, label = "mixed", "Elevated"
    else:
        status, label = "healthy", "Calm"
    shape = "near-term fear above longer-term: stressed" if term >= 1 else "normal"
    return {
        "id": "vix",
        "title": "VIX fear gauge",
        "value": round(v, 1),
        "unit": "",
        "value_label": f"VIX ÷ VIX3M {term:.2f} ({shape})",
        "status": status,
        "status_label": label,
        "counted": False,
        "what": "Expected S&P 500 volatility. Under 20 is calm, over 30 is fear. When VIX rises above VIX3M, traders are stressed right now. Sentiment, not breadth, so it's not part of the verdict.",
        "series": series(vix),
        "ref": 20,
    }


def stockbee_monitor(close, volume, universe):
    """Counts in the style of Stockbee's Market Monitor, over the wider universe."""
    cols = [t for t in universe if t in close]
    px, vol = close[cols], volume[cols]
    px = px.where(px >= 3)  # Stockbee ignores very low-priced stocks

    day = px / px.shift(1)
    vol_ok = (vol > vol.shift(1)) & (vol >= 100_000)
    up4 = ((day >= 1.04) & vol_ok).sum(axis=1)
    dn4 = ((day <= 0.96) & vol_ok).sum(axis=1)

    def ratio(n):
        return up4.rolling(n).sum() / dn4.rolling(n).sum().replace(0, np.nan)

    def count(lookback, cond):
        chg = px / px.shift(lookback)
        return cond(chg).sum(axis=1)

    m = pd.DataFrame(
        {
            "up4": up4,
            "dn4": dn4,
            "r5": ratio(5),
            "r10": ratio(10),
            "up25q": count(65, lambda c: c >= 1.25),
            "dn25q": count(65, lambda c: c <= 0.75),
            "up25m": count(20, lambda c: c >= 1.25),
            "dn25m": count(20, lambda c: c <= 0.75),
            "up50m": count(20, lambda c: c >= 1.5),
            "dn50m": count(20, lambda c: c <= 0.5),
            "up13": count(34, lambda c: c >= 1.13),
            "dn13": count(34, lambda c: c <= 0.87),
            "t2108": pct_above(px, 40),
        }
    ).iloc[-MONITOR_DAYS:][::-1]

    rows = []
    for d, r in m.iterrows():
        row = {"date": d.strftime("%Y-%m-%d")}
        for k, v in r.items():
            row[k] = None if pd.isna(v) else (round(float(v), 2) if k in ("r5", "r10") else round(float(v), 1) if k == "t2108" else int(v))
        rows.append(row)

    last = rows[0]
    if last["up25q"] > last["dn25q"] * 1.2:
        regime, text = "healthy", "More stocks are up 25%+ this quarter than down 25%+: a breakout-friendly market."
    elif last["dn25q"] > last["up25q"] * 1.2:
        regime, text = "weak", "More stocks are down 25%+ this quarter than up 25%+: breakouts tend to fail here."
    else:
        regime, text = "mixed", "Quarterly gainers and losers are roughly balanced: no strong trend either way."
    return {"universe": len(cols), "regime": regime, "regime_text": text, "rows": rows}


def optional(fn, *args):
    """Extras shouldn't block the core checks if their data is missing."""
    try:
        return [fn(*args)]
    except (KeyError, IndexError, ZeroDivisionError) as exc:
        print(f"Skipping {fn.__name__}: {exc!r}", file=sys.stderr)
        return []


def build(close, volume, sp500, universe):
    spy = close["SPY"]
    dates = [d.strftime("%Y-%m-%d") for d in close.index[-KEEP_DAYS:]]
    n500, checks = breadth_checks(close, sp500)

    score = sum({"healthy": 1, "weak": -1}.get(i["status"], 0) for i in checks)
    overall = "healthy" if score >= 2 else "weak" if score <= -2 else "mixed"
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "asof": dates[-1],
        "universe": n500,
        "overall": overall,
        "healthy_count": sum(i["status"] == "healthy" for i in checks),
        "total": len(checks),
        "spy_close": round(float(spy.iloc[-1]), 2),
        "spy_3m": round(float((spy.iloc[-1] / spy.iloc[-64] - 1) * 100), 1),
        "dates": dates,
        "indicators": checks + optional(vix_card, close),
        "monitor": stockbee_monitor(close, volume, universe),
    }


def main():
    if "--demo" in sys.argv:
        close, volume, sp500, universe = demo_data()
    else:
        sp500, universe = load_universe()
        print(f"{len(sp500)} S&P 500 / {len(universe)} total tickers", file=sys.stderr)
        close, volume = download(universe + EXTRAS)
        have = close.iloc[-1].notna()
        got500 = sum(have.get(t, False) for t in sp500)
        if got500 < 400 or not all(have.get(t, False) for t in ("SPY", "RSP")):
            sys.exit(f"Only got {got500} S&P 500 prices; refusing to overwrite data")
    data = build(close, volume, sp500, universe)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {OUT} (as of {data['asof']}, overall {data['overall']})", file=sys.stderr)


if __name__ == "__main__":
    main()
