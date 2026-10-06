import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

import argparse
import itertools
import warnings
import os

import buhl as db

# relative to this script, so it works from any working directory
out_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                            'data_files', 'sector_pairs.dat')

sectors = [
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

dt = 1./252             # one trading day, in years
min_obs = 2*252         # need at least ~2 years of overlapping data per pair
max_half_life = 63      # only plot spreads reverting within ~1 quarter (days)

sector_pairs = {}

start = "2015-01-01"
end = "2023-12-31"


# loop over sectors
for j, key in enumerate(sectors):
    sector = yf.Sector(key)
    sector_pairs[key] = []
    tickers = []
    companies = sector.top_companies
    tickers += list(companies.index[:])

    # drop duplicates / non-string entries 
    sector_tickers = [ticker for ticker in set(tickers) if isinstance(ticker,
        str) and ticker] 

    sector_prices = yf.download(sector_tickers, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
    for ticker in sector_tickers:
        sector_prices[ticker] = np.log(np.array(sector_prices[ticker]))

    nkeys = len(sector_tickers)
    ticker_pairs = [(sector_tickers[i], sector_tickers[j]) for i in range(nkeys) \
        for j in range(i+1, nkeys)]


    for i, pair in enumerate(ticker_pairs): 
        key1, key2 = pair

        p1 = sector_prices[key1]
        p2 = sector_prices[key2]
        spread, hedge_ratio, fit_info, coint_pass = db.fit_spread(p1, p2)
        nt = len(spread)
        alpha, _, pvalue = fit_info

        if coint_pass: 
            print("{} and {} are cointegratred".format(key1, key2))
            theta, mu, sigma = db.fit_SDE(spread, nt, dt)
            half_life = np.log(2)/theta
            if half_life < max_half_life:
                sector_pairs[key].append((pair, [alpha, hedge_ratio, pvalue,
                    theta, mu, sigma, half_life]))


# write output files
db.write_pairs(out_filename, sector_pairs, start, end)
print("Saved pairs to {}".format(out_filename))
