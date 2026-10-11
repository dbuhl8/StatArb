import numpy as np
import yfinance as yf
import buhl as db
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.tsa.stattools import adfuller
import os

# relative to this script, so it works from any working directory
in_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                            'data_files', 'sector_pairs.dat')
out_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                            'data_files', 'kalman_data.dat')

dt = 1./252             # one trading day, in years
sigma_alpha = 0.03      # random walk coefficients of alpha, beta (1/sqrt(year))
sigma_beta = 0.01

start = "2015-01-01"
end = "2023-12-31"

data = db.read_pairs(in_filename)

num_pairs_tested = 11770
num_pairs_found = len(data)

# downloading all prices
tickers = set([x[1] for x in data])
tickers.update([x[2] for x in data])

ticker_prices = yf.download(tickers,start=start, end=end,
        auto_adjust=True,progress=False)["Close"]

log_prices = {ticker: np.log(np.array(ticker_prices[ticker])) for ticker in
    tickers}

fig, ax = plt.subplots()
mean_improvement = 0

# one color per plotted pair, spread across a colormap as in problem 2
plot_every = 200
num_plotted = int(np.ceil(num_pairs_found/plot_every))
cmap = plt.get_cmap('plasma')
rgba_colors = cmap(np.linspace(0, 1, num_plotted))
legend_handles = []
legend_labels = []

# kalman state at the end of the formation period and stats of the kalman
# spread, saved for the backtest in problem 5
kalman_pairs = {}

for i, pair_info in enumerate(data):
    # test the kalman spread against the static spread
    # is the kalman spread closer to stationary
    sector,t1,t2,alpha,hr,adf_p,theta,mu,sigma,hl,stat_sigma,_,_ = pair_info
   
    lPa = log_prices[t1]
    lPb = log_prices[t2]

    mask = np.isfinite(lPa) & np.isfinite(lPb)
    nt = np.sum(mask)
    t = np.linspace(0, nt*dt,nt)
   
    lPa = lPa[mask]
    lPb = lPb[mask]

    kalman_spread = db.kalman_spread(lPa, lPb, dt, sigma_alpha, sigma_beta)
    alpha_k, hr_k, spread_k, P_k, R_k, lPb_mean_k = kalman_spread

    fnt = int(nt/4)
    kalman_p = adfuller(spread_k[fnt:], result_object=True)[1]
    mean_improvement += kalman_p/adf_p

    # OU fit of the kalman spread over the filtered window
    theta_k, mu_k, sigma_k = db.fit_SDE(spread_k[fnt:], nt-fnt, dt)
    hl_k = np.log(2)/(theta_k*dt)
    stat_k = sigma_k/np.sqrt(2*theta_k)

    if sector not in kalman_pairs:
        kalman_pairs[sector] = []
    kalman_pairs[sector].append(((t1, t2), [alpha_k[-1], hr_k[-1], P_k[0,0],
        P_k[0,1], P_k[1,1], R_k, lPb_mean_k, sigma_alpha, sigma_beta,
        kalman_p, theta_k, mu_k, sigma_k, hl_k, stat_k]))
    if (i%plot_every == 0):
        color = rgba_colors[i//plot_every]
        ax.plot(t, spread_k+int(i/plot_every), '-', color=color, linewidth=0.5)
        ax.plot(t, lPa-alpha-hr*lPb+int(i/plot_every), '--', color=color,
            linewidth=0.5)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=1.5))
        legend_labels.append('{}/{}'.format(t1,t2))

print('Mean(ADF Stat of Kalman Series / Static Series): {}'.\
    format(mean_improvement/num_pairs_found))
ax.set_xlabel('t')
ax.set_ylabel('Spread')

# legend is filled column by column in 2 rows, pad the pair entries to an even
# count so the line style key gets its own column
if (num_plotted%2 == 1):
    legend_handles.append(Line2D([], [], linestyle='none'))
    legend_labels.append('')

# line style key: solid = kalman, dashed = static
legend_handles += [
    Line2D([0], [0], color='k', linestyle='-', linewidth=1.5),
    Line2D([0], [0], color='k', linestyle='--', linewidth=1.5),
]
legend_labels += ['Kalman', 'Static']

fig.subplots_adjust(bottom=0.22)
fig.legend(legend_handles, legend_labels,
           loc='lower center', ncol=len(legend_handles)//2,
           bbox_to_anchor=(0.5, 0.0),
           framealpha=0.95, edgecolor='gray',
           handlelength=1.5, handletextpad=0.5, columnspacing=1.0)

fig.savefig('../pngs/problem4.png', dpi=500, bbox_inches='tight')

# write output files
db.write_pairs(out_filename, kalman_pairs, start, end,
    columns=db.kalman_columns)
print("Saved kalman data to {}".format(out_filename))
