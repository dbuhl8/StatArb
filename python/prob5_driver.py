import numpy as np
import buhl as db
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import yfinance as yf
import os


# open pairs found from problems 3/4
pairs_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                              'data_files', 'sector_pairs.dat')
pairs = db.read_pairs(pairs_filename)

# import the spreads
ticker_a = pairs["ticker_a"]
ticker_b = pairs["ticker_b"]
hedge_ratio = pairs["hedge_ratio"]
theta = pairs["theta"]
mu = pairs["mu"]
sigma = paids["sigma"]

num_pairs = len(ticker_a)

# for the test set
start = "2023-12-31"
end = "2026-10-01"


tickers = set(ticker_a+ticker_b)
prices = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=False)["Close"]

for ticker in sector_tickers:
    prices[ticker] = np.log(np.array(prices[ticker]))

# test the trading rule on testing range of the data
for i in range(num_pairs):
    spread = prices[ticker_a[i]] - hedge_ratio[i]*prices[ticker_b[i]]

    # do we want to adapt the mu, theta, sigma and the hedge_ratio with new
    # timeseries data? 
    # it would be nice to trade either version and compare the results
    # we will have an adaptive branch and stationary branch compete


# compute Sharpe Ratio, drawdown, turnover, num trades (before and after
# transaction costs)



# find broken pairs
