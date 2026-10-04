import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

import buhl as db

""" Sweep over yfinance sectors and test every within-sector pair for
cointegration using buhl.fit_spread (Engle-Granger + ADF). Pairs that pass
are fit to an OU process with buhl.fit_SDE.

Any ticker/sector that is missing, delisted, too short, or otherwise broken
is skipped with a message instead of stopping the sweep. """
""" ------------------------------------------------------------------------"""

SECTORS = [
    "basic-materials",
    "communication-services",
    "consumer-cyclical",
    "consumer-defensive",
    "energy",
    "financial-services",
    "healthcare",
    "industrials",
    "real-estate",
    "technology",
    "utilities",
]

DT = 1./252             # one trading day, in years
MIN_OBS = 2*252         # need at least ~2 years of overlapping data per pair
MAX_HALF_LIFE = 63      # only plot spreads reverting within ~1 quarter (days)
MAX_LINES_PER_PLOT = 5


def warn(msg):
    print("  [skip] {}".format(msg))


def get_sector_tickers(key, universe, n_companies):
    # returns the tickers listed for a sector, or [] if the lookup fails
    try:
        sector = yf.Sector(key)
    except Exception as e:
        warn("sector '{}' unavailable ({})".format(key, e))
        return []

    tickers = []
    if universe in ("etfs", "both"):
        try:
            etfs = sector.top_etfs
            if etfs:
                tickers += list(etfs.keys())
            else:
                warn("sector '{}' has no top ETFs listed".format(key))
        except Exception as e:
            warn("could not read ETFs for '{}' ({})".format(key, e))

    if universe in ("companies", "both"):
        try:
            companies = sector.top_companies
            if companies is not None and not companies.empty:
                tickers += list(companies.index[:n_companies])
            else:
                warn("sector '{}' has no top companies listed".format(key))
        except Exception as e:
            warn("could not read companies for '{}' ({})".format(key, e))

    # drop duplicates / non-string entries, preserving order
    seen = set()
    return [t for t in tickers if isinstance(t, str) and t
            and not (t in seen or seen.add(t))]


def download_log_prices(tickers, start, end):
    # downloads closing prices and returns a DataFrame of log prices
    # containing only the tickers with usable data
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=False)
    except Exception as e:
        warn("download failed ({})".format(e))
        return pd.DataFrame()

    if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
        warn("download returned no price data")
        return pd.DataFrame()

    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])

    good = {}
    for key in tickers:
        if key not in close.columns:
            warn("{}: no timeseries returned".format(key))
            continue
        series = pd.to_numeric(close[key], errors="coerce")
        # log of a non-positive price is undefined, treat it as missing
        series = series.where(series > 0)
        nvalid = int(series.notna().sum())
        if nvalid == 0:
            warn("{}: timeseries is empty / unavailable".format(key))
            continue
        if nvalid < MIN_OBS:
            warn("{}: only {} observations (< {})".format(key, nvalid, MIN_OBS))
            continue
        good[key] = np.log(series)

    return pd.DataFrame(good)


def test_pair(la, lb):
    # aligns two log-price series on common dates and tests for
    # cointegration. Returns a dict of results, or None if the pair is
    # unusable or not cointegrated.
    pair = pd.concat([la, lb], axis=1, join="inner").dropna()
    nt = len(pair)
    if nt < MIN_OBS:
        return None

    lPa = pair.iloc[:, 0].to_numpy()
    lPb = pair.iloc[:, 1].to_numpy()
    if np.std(lPa) == 0 or np.std(lPb) == 0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            spread, hedge_ratio, coint_pass = db.fit_spread(lPa, lPb)
        except Exception:
            return None
        if not coint_pass:
            return None

        try:
            pvalue = adfuller(spread)[1]
            theta, mu, sigma = db.fit_SDE(spread, nt, DT)
        except Exception:
            return None

    # fit_SDE returns res as an array (empty if lstsq is rank deficient)
    sigma = float(np.squeeze(sigma)) if np.size(sigma) == 1 else np.nan
    # theta is nan/negative if the AR(1) slope is outside (0,1), i.e. the
    # spread isn't actually mean reverting at the daily scale
    if not np.isfinite(theta) or theta <= 0:
        return None

    return {
        "n_obs": nt,
        "hedge_ratio": hedge_ratio,
        "adf_pvalue": pvalue,
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "half_life_days": np.log(2)/(theta*DT),
        "spread": pd.Series(spread, index=pair.index),
    }


def sweep_sector(key, universe, n_companies, start, end):
    print("\n=== {} ===".format(key))
    tickers = get_sector_tickers(key, universe, n_companies)
    if len(tickers) < 2:
        warn("fewer than 2 tickers for '{}'".format(key))
        return []
    print("  tickers: {}".format(", ".join(tickers)))

    log_px = download_log_prices(tickers, start, end)
    if log_px.shape[1] < 2:
        warn("fewer than 2 usable timeseries for '{}'".format(key))
        return []

    results = []
    pairs = list(itertools.combinations(log_px.columns, 2))
    for key1, key2 in pairs:
        res = test_pair(log_px[key1], log_px[key2])
        if res is None:
            continue
        res.update({"sector": key, "ticker_a": key1, "ticker_b": key2})
        results.append(res)
        print("  {} & {} are cointegrated (p = {:.3g}, half-life = {:.1f} d)"
              .format(key1, key2, res["adf_pvalue"], res["half_life_days"]))

    print("  {}/{} pairs cointegrated".format(len(results), len(pairs)))
    return results


def plot_spreads(results, fname):
    by_sector = {}
    for r in results:
        if r["half_life_days"] <= MAX_HALF_LIFE:
            by_sector.setdefault(r["sector"], []).append(r)
    if not by_sector:
        print("\nNo pairs with half-life <= {} days to plot".format(MAX_HALF_LIFE))
        return

    nplots = len(by_sector)
    fig, ax = plt.subplots(nplots, 1, figsize=(8, 2.5*nplots), squeeze=False)
    for i, (sector, rs) in enumerate(by_sector.items()):
        rs = sorted(rs, key=lambda r: r["adf_pvalue"])[:MAX_LINES_PER_PLOT]
        for r in rs:
            ax[i, 0].plot(r["spread"], linewidth=0.7,
                          label="{}&{}".format(r["ticker_a"], r["ticker_b"]))
        ax[i, 0].set_title(sector)
        ax[i, 0].set_ylabel('S')
        ax[i, 0].legend(fontsize=6, loc="upper left")
    ax[-1, 0].set_xlabel('t')
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    print("\nSaved spread plot to {}".format(fname))


def main():
    parser = argparse.ArgumentParser(description=
        "Sweep yfinance sectors for cointegrated pairs.")
    parser.add_argument("--sectors", nargs="+", default=SECTORS,
                        help="yfinance sector keys to sweep")
    parser.add_argument("--universe", choices=["etfs", "companies", "both"],
                        default="etfs", help="which tickers to pull per sector")
    parser.add_argument("--n-companies", type=int, default=10,
                        help="number of top companies per sector")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--csv", default="sector_sweep_pairs.csv")
    parser.add_argument("--plot", default="sector_sweep_spreads.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    results = []
    for key in args.sectors:
        # last line of defence: one bad sector shouldn't kill the sweep
        try:
            results += sweep_sector(key, args.universe, args.n_companies,
                                    args.start, args.end)
        except Exception as e:
            warn("sector '{}' failed unexpectedly ({})".format(key, e))

    if not results:
        print("\nNo cointegrated pairs found.")
        return

    table = pd.DataFrame([{k: v for k, v in r.items() if k != "spread"}
                          for r in results])
    table = table[["sector", "ticker_a", "ticker_b", "n_obs", "hedge_ratio",
                   "adf_pvalue", "theta", "mu", "sigma", "half_life_days"]]
    table = table.sort_values(["sector", "adf_pvalue"]).reset_index(drop=True)
    table.to_csv(args.csv, index=False)

    print("\n{} cointegrated pairs found, saved to {}".format(len(table), args.csv))
    with pd.option_context("display.width", 120, "display.max_rows", 200):
        print(table.to_string(float_format=lambda x: "{:.4g}".format(x)))

    if not args.no_plot:
        plot_spreads(results, args.plot)


if __name__ == "__main__":
    main()
