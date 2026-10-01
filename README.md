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

The headline verdict counts the checks: two or more net healthy is "Broad", two or more net weak is "Narrow / weak", and anything else is "Mixed".

## Data sources (free)

- S&P 500 members: Wikipedia, with a fallback to the [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies) list.
- Daily prices for every member, plus SPY and RSP: Yahoo Finance, fetched with [`yfinance`](https://github.com/ranaroussi/yfinance).

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
