import numpy as np
import yfinance as yf
import buhl as db
import os

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
pairs_tested = 0
for j, key in enumerate(sectors):
    sector = yf.Sector(key)
    sector_pairs[key] = []
    tickers = []
    companies = sector.top_companies
    tickers += list(companies.index[:])

    # drop duplicates / non-string entries 
    sector_tickers = [ticker for ticker in set(tickers) if isinstance(ticker,
        str) and ticker] 

    raw_prices = yf.download(sector_tickers,start=start, end=end,
        auto_adjust=True,progress=False)["Close"]

    # plain numpy arrays keyed by ticker, all on the same dates
    sector_prices = {ticker: np.log(np.array(raw_prices[ticker])) for ticker in
        sector_tickers}

    bad_tickers = []

    for ticker in sector_tickers:
        if (np.sum(~np.isnan(sector_prices[ticker])) < min_obs):
            bad_tickers.append(ticker)
    
    sector_tickers = [x for x in sector_tickers if x not in bad_tickers]
    
    nkeys = len(sector_tickers)
    ticker_pairs = [(sector_tickers[i], sector_tickers[j]) for i in range(nkeys) \
        for j in range(i+1, nkeys)]

    pairs_tested += len(ticker_pairs)
    for i, pair in enumerate(ticker_pairs):
        key1, key2 = pair
        print('Testing pair: ({}, {})'.format(key1, key2))
        p1 = sector_prices[key1]
        p2 = sector_prices[key2]
        # check if the pair together has enough data for cointegration tests
        mask = np.isfinite(p1) & np.isfinite(p2)
        if (np.sum(mask) < min_obs):
            continue
        else: 
            p1 = p1[mask]
            p2 = p2[mask]
        spread, hedge_ratio, fit_info, coint_pass = db.fit_spread(p1, p2)
        nt = len(spread)
        alpha, _, pvalue = fit_info
        if coint_pass: 
            print("{} and {} are cointegratred".format(key1, key2))
            theta, mu, sigma = db.fit_SDE(spread, nt, dt)
            if (theta > 0):
                stat_std_dev = sigma/np.sqrt(2*theta)
                # division by dt converts from years to days 
                half_life = np.log(2)/(theta*dt)
                if half_life < max_half_life:
                    sector_pairs[key].append((pair, [alpha, hedge_ratio, pvalue,
                        theta, mu, sigma, half_life,stat_std_dev]))


# write output files
db.write_pairs(out_filename, sector_pairs, start, end)
print('Number of pairs tested: {}'.format(pairs_tested))
print("Saved pairs to {}".format(out_filename))
