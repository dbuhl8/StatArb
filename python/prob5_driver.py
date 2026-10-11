import numpy as np
import buhl as db
import matplotlib.pyplot as plt
import yfinance as yf
import os


# open pairs found from problems 3/4, both files have one row per pair in the
# same order, so row i is the same pair in each
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                        'data_files')
pairs = db.read_pairs(os.path.join(data_dir, 'sector_pairs.dat'))
kalman = db.read_pairs(os.path.join(data_dir, 'kalman_data.dat'))

if not (np.all(pairs["ticker_a"] == kalman["ticker_a"]) and
        np.all(pairs["ticker_b"] == kalman["ticker_b"])):
    raise ValueError('sector_pairs.dat and kalman_data.dat pairs do not match, '
                     'rerun prob4_driver.py')

ticker_a = pairs["ticker_a"]
ticker_b = pairs["ticker_b"]
num_pairs = len(ticker_a)

# for the test set, the formation period ends 2023-12-31
start = "2024-01-01"
end = "2026-10-01"

dt = 1./252             # one trading day, in years
z_in = 2.               # enter when |z| > z_in
z_out = 1.              # exit when |z| < z_out
lag = 10                # days used for the trend filter

# downloading all prices
tickers = sorted(set(ticker_a) | set(ticker_b))
ticker_prices = yf.download(tickers, start=start, end=end, auto_adjust=True,
                            progress=False)["Close"]
dates = ticker_prices.index
nt_all = len(dates)

# plain numpy arrays keyed by ticker, all on the same dates
prices = {ticker: np.array(ticker_prices[ticker]) for ticker in tickers}
log_prices = {ticker: np.log(prices[ticker]) for ticker in tickers}

# the four strategies being compared: (spread, trend filter on/off)
variants = [('static', False), ('static', True), ('kalman', False),
            ('kalman', True)]

# inventory: shares held at the close of each day, per pair, on the full test
# date grid, so the daily pnl is shares[t-1]*(P[t] - P[t-1]), with prices
# forward filled over missing days
shares_a = {v: np.zeros((num_pairs, nt_all)) for v in variants}
shares_b = {v: np.zeros((num_pairs, nt_all)) for v in variants}

# ledger: one row per leg per transaction, shares > 0 is a purchase and
# shares < 0 a sale, costs and turnover are computed from this afterwards
ledger_rows = []

# test the trading rule on testing range of the data
for i in range(num_pairs):
    t1, t2 = ticker_a[i], ticker_b[i]

    # only days where both prices exist, mapped back to the full grid by days
    mask = np.isfinite(prices[t1]) & np.isfinite(prices[t2])
    days = np.where(mask)[0]
    nt = days.size
    if (nt < lag+2):
        print('Pair ({}, {}) has too little test data, skipping'.format(t1, t2))
        continue

    Pa, Pb = prices[t1][mask], prices[t2][mask]
    lPa, lPb = log_prices[t1][mask], log_prices[t2][mask]

    # static spread, all parameters from the formation period
    static_spread = lPa - pairs["alpha"][i] - pairs["hedge_ratio"][i]*lPb
    z_static = (static_spread - pairs["mu"][i])/pairs["stat_std_dev"][i]
    hr_static = np.ones(nt)*pairs["hedge_ratio"][i]

    # kalman spread, continue the filter from its state at the end of the
    # formation period, spread_k[t] only uses alpha, beta from t-1
    P0 = np.array([[kalman["P00"][i], kalman["P01"][i]],
                   [kalman["P01"][i], kalman["P11"][i]]])
    alpha_k, hr_k, spread_k, P_k, R_k, lPb_mean_k = db.kalman_spread(lPa, lPb,
        dt, kalman["sigma_alpha"][i], kalman["sigma_beta"][i],
        alpha0=kalman["alpha"][i], beta0=kalman["hedge_ratio"][i], P0=P0,
        R=kalman["R"][i], lPb_mean=kalman["lPb_mean"][i])
    z_kalman = (spread_k - kalman["mu"][i])/kalman["stat_std_dev"][i]

    signals = {'static': (z_static, hr_static), 'kalman': (z_kalman, hr_k)}

    # a missing price in the middle of the window doesn't close the position,
    # so on the full grid each day holds what was held on the last valid day
    full_range = np.arange(days[0], days[-1]+1)
    last_valid = np.searchsorted(days, full_range, side='right') - 1

    for v in variants:
        branch, trend_filter = v
        z, hr = signals[branch]
        sa, sb, trades = db.pairs_backtest(Pa, Pb, lPa, lPb, z, hr, z_in=z_in,
            z_out=z_out, lag=lag, dt=dt, trend_filter=trend_filter)
        shares_a[v][i, full_range] = sa[last_valid]
        shares_b[v][i, full_range] = sb[last_valid]
        for day, leg, shares, price in trades:
            ticker = t1 if (leg == 'A') else t2
            ledger_rows.append((i, days[day], leg, ticker, shares, price,
                                branch, trend_filter))

ledger = np.array(ledger_rows, dtype=[('pair', int), ('day', int),
    ('leg', 'U1'), ('ticker', 'U8'), ('shares', float), ('price', float),
    ('branch', 'U6'), ('trend_filter', bool)])
purchases = ledger[ledger['shares'] > 0]
sales = ledger[ledger['shares'] < 0]

for v in variants:
    in_v = (ledger['branch'] == v[0]) & (ledger['trend_filter'] == v[1])
    # each round trip is 4 ledger rows, entry and exit of both legs
    print('{:6s} trend filter {:5s}: {} trades'.format(v[0], str(v[1]),
        np.sum(in_v)//4))

# compute Sharpe Ratio, drawdown, turnover, num trades (before and after
# transaction costs)



# find broken pairs
