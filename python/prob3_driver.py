import numpy as np
import buhl as db
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import yfinance as yf

""" Testing Cointegration method """ 
""" ------------------------------------------------------------------------"""

px = yf.download(["KO", "PEP"], start="2015-01-01", end="2023-12-31",
                 auto_adjust=True)["Close"]   
KO = np.array(px["KO"])
PEP = np.array(px["PEP"])

px_test = yf.download(["KO", "PEP"], start="2024-01-01", end="2026-9-25",
                 auto_adjust=True)["Close"]   
KO_test = np.array(px_test["KO"])
PEP_test = np.array(px_test["PEP"])


lKO = np.log(KO)
lPEP = np.log(PEP)

spread, hedge_ratio, coint_pass = db.fit_spread(lKO, lPEP)

# Forming the spread
dt = 0.01
nt = int(spread.size)

A = np.stack((spread[:nt-1],np.exp(np.zeros_like(spread[:nt-1]))), axis=1)
coeff, res, _, _ = np.linalg.lstsq(A,spread[1:])
m, b = coeff

# Derive properties of the SDE from the interpolation
theta = -np.log(m)/dt
mu = b/(1. - m)
sigma = np.sqrt(res/nt)

fig,ax = plt.subplots()

t = np.linspace(0, nt*dt, nt)

ax.plot(t, spread) 
ax.set_xlabel('t')
ax.set_ylabel('Spread')

fig.savefig('problem3.png', dpi=500)


""" Testing top EFTS in a sector""" 
""" ------------------------------------------------------------------------"""

fig2, ax2 = plt.subplots()

sector = yf.Sector("utilities")
s_ETF_keys = list(sector.top_etfs.keys())
s_ETF_keys.remove("UTES")
s_ETF_keys.remove("ZAP")
nkeys = len(s_ETF_keys)

sector_data = yf.download(s_ETF_keys, start="2015-01-01", end="2023-12-31",
    auto_adjust=True)["Close"]

for key in s_ETF_keys:
    sector_data[key] = np.log(np.array(sector_data[key]))

etf_pairs = [(s_ETF_keys[i], s_ETF_keys[j]) for i in range(nkeys) \
    for j in range(i+1, nkeys)]

for i, pair in enumerate(etf_pairs): 
    print(pair)
    key1, key2 = pair
    # testing etf pairs
    etf1 = sector_data[key1]
    etf2 = sector_data[key2]
    spread, hedge_ratio, coint_pass = db.fit_spread(etf1, etf2)

    if coint_pass: 
        print("ETFS {} and {} are cointegratred".format(key1, key2))
        theta, mu, sigma = db.fit_SDE(spread, nt, dt)
        half_life = np.log(2/theta)
        if half_life < nt*dt*0.1:
            ax2.plot(t,spread,label="{}&{}".format(key1,key2))


ax2.set_xlabel('t')
ax2.set_ylabel('S')
ax2.legend()
ax2.set_title('Utilities Sector Cointegrated ETF Spreads')

fig2.savefig('problem3_etf_spreads.png', dpi=500)


         

