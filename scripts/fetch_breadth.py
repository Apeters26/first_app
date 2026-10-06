"""Build data/breadth.json: market-breadth indicators, a VIX gauge and a Stockbee-style monitor.

Free sources only:
  - S&P 500 / 400 / 600 member lists: Wikipedia (S&P 500 fallback: datasets/s-and-p-500-companies)
  - Daily prices and volume: Yahoo Finance via the yfinance package

The five breadth checks use the S&P 500. The Stockbee-style monitor uses the
wider S&P 1500 (500 + 400 + 600) so the big-mover counts are meaningful.

Run:  python scripts/fetch_breadth.py            (live data)
      python scripts/fetch_breadth.py --demo     (synthetic data, for offline testing)
"""

import csv
import io
import json
import re
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
INDEXES = [
    ("^GSPC", "S&P 500"),
    ("^IXIC", "Nasdaq Composite"),
    ("^NDX", "Nasdaq 100"),
    ("^DJI", "Dow Jones"),
    ("^RUT", "Russell 2000"),
]
EXTRAS = ["SPY", "RSP", "^VIX", "^VIX3M"] + [t for t, _ in INDEXES]
TOP_N = 25  # largest companies shown in the moving-average table
# (key, label, moving average, lookback in days used to judge its slope)
MA_SPECS = [
    ("e21", "21-day EMA", lambda p: p.ewm(span=21, adjust=False).mean(), 5),
    ("s50", "50-day", lambda p: p.rolling(50).mean(), 10),
    ("s200", "200-day", lambda p: p.rolling(200).mean(), 20),
]
FLAT_BAND = 0.1  # % change in the average over its lookback that still counts as flat
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_{}_companies"
# Stockbee's public Market Monitor sheet (he keeps updating this one; the title says 2025).
STOCKBEE_SHEET = "https://docs.google.com/spreadsheets/d/1O6OhS7ciA8zwfycBfGPbP2fWJnR0pn2UUvFZVDP9jpE"


def wiki_symbols(n):
    html = requests.get(WIKI.format(n), headers=UA, timeout=30).text
    for table in pd.read_html(io.StringIO(html)):
        for col in table.columns:
            if str(col).lower() in ("symbol", "ticker symbol", "ticker"):
                name_col = next((c for c in table.columns if str(c).lower() in ("security", "company")), col)
                return {
                    str(sym).strip().replace(".", "-"): str(name).strip()
                    for sym, name in zip(table[col], table[name_col])
                    if pd.notna(sym)
                }
    raise ValueError(f"no symbol column on S&P {n} page")


def load_universe():
    try:
        sp500 = wiki_symbols(500)
    except Exception as exc:  # noqa: BLE001
        print(f"Wikipedia S&P 500 failed ({exc}); using GitHub dataset", file=sys.stderr)
        text = requests.get(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
            headers=UA,
            timeout=30,
        ).text
        table = pd.read_csv(io.StringIO(text))
        name_col = "Security" if "Security" in table else "Name" if "Name" in table else "Symbol"
        sp500 = {str(s).strip().replace(".", "-"): str(n) for s, n in zip(table["Symbol"], table[name_col])}
    wider = set(sp500)
    for n in (400, 600):
        try:
            wider |= set(wiki_symbols(n))
        except Exception as exc:  # noqa: BLE001
            print(f"S&P {n} list failed ({exc}); monitor uses a smaller universe", file=sys.stderr)
    return sp500, sorted(wider)


def download(tickers):
    import yfinance as yf

    data = yf.download(
        tickers, period="2y", interval="1d", auto_adjust=True, progress=False, threads=True
    )
    close, volume = data["Close"].dropna(how="all"), data["Volume"]
    # Keep only stock-market trading days; ^VIX can print on exchange holidays,
    # which would leave a gap in every stock's history.
    close = close[close["SPY"].notna()]
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
    for k, (sym, _) in enumerate(INDEXES):
        close[sym] = (1000 * (k + 1)) * np.exp(np.cumsum(market * rng.uniform(0.8, 1.4) + rng.normal(0, 0.003, len(dates))))
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
    ma = prices.rolling(window, min_periods=int(window * 0.9)).mean()
    valid = ma.notna() & prices.notna()
    above = ((prices > ma) & valid).sum(axis=1)
    return above / valid.sum(axis=1).replace(0, np.nan) * 100


def breadth_checks(close, sp500):
    spy, rsp = close["SPY"], close["RSP"]
    stocks = close[[t for t in sp500 if t in close]]  # works for a list or a {symbol: name} dict
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

    t2108 = pct_above(px, 40)
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
            "t2108": t2108,
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
    return {"universe": len(cols), "regime": regime, "regime_text": text, "rows": rows}, t2108


def fetch_stockbee_t2108():
    """T2108 column from Stockbee's sheet, as a date-indexed Series (newest last)."""
    text = requests.get(f"{STOCKBEE_SHEET}/pub?output=csv", headers=UA, timeout=30).text
    rows = list(csv.reader(io.StringIO(text)))
    hi = next(i for i, r in enumerate(rows) if r and r[0].strip().lower() == "date")
    header = [h.strip().lower() for h in rows[hi]]
    col = next((i for i, h in enumerate(header) if "t2108" in h), 14)  # 14: its spot in the 2026 layout
    out = {}
    for r in rows[hi + 1:]:
        try:
            d = pd.to_datetime(r[0], format="%m/%d/%Y")
            v = float(r[col].replace(",", ""))
        except (ValueError, IndexError):
            continue
        if 0 <= v <= 100:
            out[d] = v
    if len(out) < 5:
        raise ValueError(f"only {len(out)} T2108 rows parsed")
    return pd.Series(out).sort_index()


def t2108_card(dates, ours, theirs):
    asof = pd.Timestamp(dates[-1])
    fresh = theirs is not None and theirs.index[-1] >= asof - pd.Timedelta(days=7)
    v = float(theirs.iloc[-1]) if fresh else float(ours.iloc[-1])
    if v < 20:
        status, label = "weak", "Oversold"
    elif v > 70:
        status, label = "mixed", "Overbought"
    elif v < 40:
        status, label = "mixed", "Below average"
    else:
        status, label = "healthy", "Healthy"
    idx = pd.to_datetime(dates)
    ours_txt = f"S&P 1500 version {ours.iloc[-1]:.1f}"
    if fresh:
        when = theirs.index[-1].strftime("%b %-d")
        value_label = f"Stockbee, {when} · {ours_txt}"
        line = series(theirs.reindex(idx))
    else:
        value_label = f"{ours_txt} (Stockbee sheet unavailable)"
        line = series(ours.reindex(idx))
    return {
        "id": "t2108",
        "title": "T2108",
        "value": round(v, 1),
        "unit": "%",
        "value_label": value_label,
        "status": status,
        "status_label": label,
        "counted": False,
        "what": "Share of stocks above their 40-day average. Under 20 is a bottoming zone; over 70 is overheated. Stockbee's number covers about 6,500 stocks; ours covers the S&P 1500.",
        "series": line,
        "series2": series(ours.reindex(idx)) if fresh else None,
        "labels": ["Stockbee", "S&P 1500 (ours)"] if fresh else None,
        "refs": [20, 70],
        "source": f"{STOCKBEE_SHEET}/pubhtml" if fresh else None,
    }


def ma_profile(price):
    """Distance from, and slope of, the 21-day EMA and 50/200-day averages."""
    p = price.dropna()
    if len(p) < 30:
        raise ValueError("not enough history")
    last = float(p.iloc[-1])
    out = {
        "price": round(last, 2),
        "chg1d": round((last / p.iloc[-2] - 1) * 100, 2),
        "chg1m": round((last / p.iloc[-22] - 1) * 100, 1),
    }
    above = rising = 0
    for key, _, fn, lookback in MA_SPECS:
        ma = fn(p)
        if len(ma.dropna()) <= lookback:
            out[key] = None
            continue
        m, prev = float(ma.iloc[-1]), float(ma.iloc[-1 - lookback])
        change = (m / prev - 1) * 100
        slope = "up" if change > FLAT_BAND else "down" if change < -FLAT_BAND else "flat"
        out[key] = {"dist": round((last / m - 1) * 100, 1), "slope": slope}
        above += last > m
        rising += slope == "up"
    n = sum(out[k] is not None for k, *_ in MA_SPECS)
    if n and above == n and rising == n:
        out["trend"] = "healthy"
    elif n and above == 0 and not any(out[k] and out[k]["slope"] != "down" for k, *_ in MA_SPECS):
        out["trend"] = "weak"
    else:
        out["trend"] = "mixed"
    return out


def index_table(close):
    rows = []
    for sym, name in INDEXES:
        if sym in close and close[sym].notna().sum() > 200:
            rows.append({"sym": sym.lstrip("^"), "name": name, **ma_profile(close[sym])})
    if not rows:
        raise KeyError("no index prices")
    return rows


def market_caps(tickers):
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor

    def cap(t):
        try:
            return t, float(yf.Ticker(t).fast_info["market_cap"])
        except Exception:  # noqa: BLE001 - one missing cap shouldn't stop the rest
            return t, None

    with ThreadPoolExecutor(8) as pool:
        return {t: c for t, c in pool.map(cap, tickers) if c}


def megacap_table(close, volume, sp500, caps_fn=market_caps):
    """The TOP_N biggest S&P 500 companies with the same moving-average profile."""
    names = sp500 if isinstance(sp500, dict) else {t: t for t in sp500}
    syms = [t for t in names if t in close]
    dollar_vol = (close[syms] * volume[syms]).iloc[-20:].mean().sort_values(ascending=False)
    candidates = list(dollar_vol.index[:60])  # every giant trades heavily; cap lookups only for these
    caps = caps_fn(candidates)
    by_cap = len(caps) >= 40
    order = sorted(caps, key=caps.get, reverse=True) if by_cap else candidates

    rows, seen = [], set()
    for t in order:
        company = re.sub(r"\s*\(?Class [A-Z]\)?$", "", names.get(t, t)).strip()
        if company in seen:  # one row per company (e.g. GOOGL and GOOG)
            continue
        try:
            prof = ma_profile(close[t])
        except ValueError:
            continue
        seen.add(company)
        rows.append({"sym": t, "name": company, "cap": round(caps[t] / 1e9) if by_cap else None, **prof})
        if len(rows) == TOP_N:
            break
    s50 = [r["s50"] for r in rows if r.get("s50")]
    return {
        "basis": "market cap" if by_cap else "trading value (market caps unavailable)",
        "above50": sum(x["dist"] > 0 for x in s50),
        "above200": sum(r["s200"]["dist"] > 0 for r in rows if r.get("s200")),
        "uptrend": sum(r["trend"] == "healthy" for r in rows),
        "downtrend": sum(r["trend"] == "weak" for r in rows),
        "rows": rows,
    }


def optional(fn, *args):
    """Extras shouldn't block the core checks if their data is missing."""
    try:
        return [fn(*args)]
    except (KeyError, IndexError, ZeroDivisionError, ValueError, StopIteration, requests.RequestException) as exc:
        print(f"Skipping {fn.__name__}: {exc!r}", file=sys.stderr)
        return []


def build(close, volume, sp500, universe, stockbee=None, caps_override=()):
    spy = close["SPY"]
    dates = [d.strftime("%Y-%m-%d") for d in close.index[-KEEP_DAYS:]]
    n500, checks = breadth_checks(close, sp500)
    monitor, our_t2108 = stockbee_monitor(close, volume, universe)

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
        "indicators": checks + optional(vix_card, close) + optional(t2108_card, dates, our_t2108, stockbee),
        "monitor": monitor,
        "indices": next(iter(optional(index_table, close)), None),
        "megacaps": next(iter(optional(megacap_table, close, volume, sp500, *caps_override)), None),
    }


def clean(obj):
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def main():
    stockbee, caps_override = None, ()
    if "--demo" in sys.argv:
        close, volume, sp500, universe = demo_data()
        sp500 = {t: f"Demo Company {t[1:]}" for t in sp500}
        caps_override = (lambda ts: {t: 4e12 / (i + 1) for i, t in enumerate(ts)},)
        idx = close.index[-120:]
        stockbee = pd.Series(np.linspace(30, 22, len(idx)), index=idx)
    else:
        sp500, universe = load_universe()
        print(f"{len(sp500)} S&P 500 / {len(universe)} total tickers", file=sys.stderr)
        close, volume = download(universe + EXTRAS)
        have = close.iloc[-1].notna()
        got500 = sum(have.get(t, False) for t in sp500)
        if got500 < 400 or not all(have.get(t, False) for t in ("SPY", "RSP")):
            sys.exit(f"Only got {got500} S&P 500 prices; refusing to overwrite data")
        got = optional(fetch_stockbee_t2108)
        stockbee = got[0] if got else None
    data = build(close, volume, sp500, universe, stockbee, caps_override)
    OUT.parent.mkdir(exist_ok=True)
    # allow_nan=False: a stray NaN would make the file unreadable in browsers.
    OUT.write_text(json.dumps(clean(data), separators=(",", ":"), allow_nan=False))
    print(f"Wrote {OUT} (as of {data['asof']}, overall {data['overall']})", file=sys.stderr)


if __name__ == "__main__":
    main()
