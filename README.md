# Market Breadth

A simple dashboard showing how many S&P 500 stocks are taking part in the market's move. It's built from free data only.

## The five checks

| Check | What it tells you | Healthy / Weak |
|---|---|---|
| % of stocks above 200-day average | Long-term participation | > 60% / < 40% |
| % of stocks above 50-day average | Short-term participation | > 60% / < 40% |
| New 52-week highs minus new lows (10-day avg) | More leaders or more laggards? | > +5 / < −5 |
| Advance/decline line + McClellan oscillator | Are most stocks rising day to day? | A/D line above its 50-day avg **and** McClellan > 0 / both negative |
| Equal-weight vs. cap-weight (RSP ÷ SPY, 3-month change) | Is the average stock keeping up with the giants? | > +1% / < −1% |

### Also on the page

- **VIX fear gauge** (`^VIX`, with `^VIX3M` for term structure). Under 20 is calm and over 30 is fear. VIX above VIX3M means near-term stress. This is sentiment, so it doesn't count toward the verdict.
- **Stockbee-style Market Monitor** (collapsed by default). It covers the last 10 days for the S&P 1500 (stocks priced $3+):
  - stocks up or down 4%+ on rising volume, with their 5- and 10-day ratios
  - stocks up or down 25% in a quarter, 25% and 50% in a month, and 13% in 34 days
  - T2108 (% of stocks above their 40-day average)

  The quarterly 25% counts set the "regime" label. Stockbee scans about 6,000 stocks, so the raw counts here are smaller than his; compare up vs. down instead.

- **Indices vs. moving averages**: the S&P 500, Nasdaq Composite, Nasdaq 100, Dow and Russell 2000. For each, the table shows the % distance from the 20-, 50- and 200-day simple moving averages (SMA), plus which way each average is sloping. A slope is judged over 5, 10 and 20 days respectively; a move within ±0.1% counts as flat. "Uptrend" means price is above all three and all three are rising.
- **Top 25 stocks by market cap** (collapsed by default): the same moving-average breakdown for the biggest S&P 500 companies, one row per company. Market caps come from Yahoo; if those lookups fail, the ranking falls back to trading value.

The headline verdict counts the five checks: two or more net healthy is "Broad", two or more net weak is "Narrow / weak", and anything else is "Mixed".

## Data sources (free)

- S&P 500, 400 and 600 members: Wikipedia, with an S&P 500 fallback to the [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies) list.
- Daily prices and volume for every member, plus SPY, RSP, ^VIX and ^VIX3M: Yahoo Finance, fetched with [`yfinance`](https://github.com/ranaroussi/yfinance).

All breadth numbers are calculated from these prices in `scripts/fetch_breadth.py`.

## How it runs

- `.github/workflows/update-breadth.yml` runs on weekdays after the close. It can also be started by hand from the Actions tab.
- The workflow regenerates `data/breadth.json` and commits it.
- `index.html` is a single static page that reads that JSON.

To publish the site, go to **Settings → Pages**, choose **Deploy from a branch**, and select `master` / `(root)`.

## Run locally

```sh
pip install -r requirements.txt
python scripts/fetch_breadth.py        # or --demo for synthetic data
python -m http.server                  # then open http://localhost:8000
```
