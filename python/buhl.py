import numpy as np
from statsmodels.tsa.stattools import adfuller

# Buhl Module File for SDEs


def euler_maruyama(x,t,A,B,dt):
    # computes the update X_n+1 = A(X_n, t_n, dt) + B(X_n, t_n, dt)dW
    # where dW ~ N(0,dt) is brownian motion
    # x, t should be the state variables
    # A, B should be the respective RHS functions
    # and dt is the timestep
    nx = np.zeros_like(x)
    dW = np.random.standard_normal()
    nx = x + A(x,t,dt)*dt + B(x,t,dt)*dW*np.sqrt(dt)
    return nx, dW


def fit_spread(lPa, lPb):
    # tests two candidate pairs for cointegration
    # and then forms the spread, if the spread isn't stationary a warning is
    # thrown
        
    # solves the linear regression problem to fit the spread
    A = np.stack((lPb,np.ones_like(lPb)), axis=1)
    coeff,resid,_,_ = np.linalg.lstsq(A,lPa)
    hedge_ratio, alpha = coeff
    S = lPa - hedge_ratio*lPb - alpha
    # Augmented Dickey-Fuller test to see if the spread is stationary
    adf_result = adfuller(S, result_object=True) # c.f. statsmodels.tsa.stattools.adfuller
    pvalue = adf_result[1]
    # need to extract the ADF result
    if pvalue >= 0.05: 
        #print('Warning: the spread does not pass the ADF test (p = {})'.format(pvalue))
        result = False
    else:
        result = True
    return S, hedge_ratio, [alpha, resid[0],pvalue], result

def kalman_spread(lPa, lPb, dt, sigma_alpha=0.03, sigma_beta=0.01,
                  alpha0=None, beta0=None, P0=None, R=None, lPb_mean=None):
    # fits a kalman filter to a potential spread
    # by default, starts by isolating a quarter of the initial timeseries,
    # fits a least-squares regression on it, and then loops forward and
    # applies the kalman filter to each following timestep
    # if alpha0, beta0, P0, R, lPb_mean are given (e.g. the state returned by
    # a previous call) the initial fit is skipped and the filter continues
    # from that state starting at the first timestep. alpha0, beta0, P0 are in
    # the original (uncentered) variables, as returned by this function
    # sigma_alpha, sigma_beta are the random walk coefficients of alpha, beta
    # i.e. d(alpha) = sigma_alpha dW, in units of 1/sqrt(year) when dt = 1/252
    # returns alpha, hedge_ratio, and the spread as timeseries, the final P,
    # R and lPb_mean, all in the original (uncentered) variables, so that
    # lPa = alpha + hedge_ratio*lPb + spread, matching fit_spread
    nt = lPa.size
    given_state = [alpha0, beta0, P0, R, lPb_mean]

    if alpha0 is None:
        fnt = int(nt/4)

        # center lPb on the initial window, this is only a change of variables:
        # alpha + beta*lPb = (alpha + beta*lPb_mean) + beta*(lPb - lPb_mean)
        # so beta and the spread are unchanged, only alpha is shifted. This
        # makes the initial alpha and beta uncorrelated so a diagonal Q makes
        # sense
        lPb_mean = np.mean(lPb[:fnt])
        x = lPb - lPb_mean

        # initial fit on reduced timeseries
        S, beta_c, other_stuff, result = fit_spread(lPa[:fnt], x[:fnt])
        alpha_c, res, pvalue = other_stuff

        # initializing kalman recursion
        # R: variance of the observation noise, residual variance of initial fit
        R = res/(fnt-2)
        # P: covariance of (alpha, beta) from the initial least-squares fit
        A = np.stack((np.ones(fnt), x[:fnt]), axis=1)
        P = R*np.linalg.inv(A.T@A)
        del A
    elif any(value is None for value in given_state):
        raise ValueError('kalman_spread: alpha0, beta0, P0, R and lPb_mean '
                         'must all be given to start from a previous state')
    else:
        fnt = 0
        S = np.zeros(0)
        # move the given state into the centered variables,
        # alpha_c = alpha + beta*lPb_mean and P_c = Tinv P Tinv^T
        x = lPb - lPb_mean
        beta_c = beta0
        alpha_c = alpha0 + beta0*lPb_mean
        Tinv = np.array([[1., lPb_mean], [0., 1.]])
        P = Tinv@np.asarray(P0, dtype=float)@Tinv.T

    alpha = np.ones_like(lPa)*alpha_c
    hedge_ratio = np.ones_like(lPa)*beta_c
    spread = np.zeros_like(lPa)
    spread[:fnt] = S

    # Q: covariance of the random walk increments of (alpha, beta) per step
    Q = np.diag([sigma_alpha**2, sigma_beta**2])*dt

    # propagating kalman filter forward, a_prev, b_prev hold alpha, beta from
    # the previous timestep (or the initial state on the first step)
    a_prev, b_prev = alpha_c, beta_c
    for i in range(fnt,nt):
        H = np.array([1., x[i]])
        # propagate the uncertainty forward
        P = P + Q
        # compute the error in the prediction, this is the spread at time i
        # using only alpha, beta known at time i-1
        err = lPa[i] - a_prev - b_prev*x[i]
        # compute uncertainty of observations
        OE = H@P@H + R
        # Kalman Gain
        K = P@H/OE
        # Update alpha, hedge_ratio, uncertainty
        alpha[i] = a_prev + K[0]*err
        hedge_ratio[i] = b_prev + K[1]*err
        P = (np.eye(2) - np.outer(K,H))@P
        spread[i] = err
        a_prev, b_prev = alpha[i], hedge_ratio[i]

    # undo the centering, alpha -> alpha - beta*lPb_mean and P -> T P T^T
    alpha = alpha - hedge_ratio*lPb_mean
    T = np.array([[1., -lPb_mean], [0., 1.]])
    P = T@P@T.T
    return [alpha, hedge_ratio, spread, P, R, lPb_mean]
         
    
def fit_SDE(spread, nt, dt):
    # given a stationary, mean-reverting timeseries
    A = np.stack((spread[:nt-1],np.ones_like(spread[:nt-1])), axis=1)
    coeff, res, _, _ = np.linalg.lstsq(A,spread[1:])
    b, a = coeff

    # Derive properties of the SDE from the interpolation
    theta = -np.log(b)/dt
    mu = a/(1. - b)
    sigma = np.sqrt(-2.*(res/nt)*np.log(b)/(dt*(1-b**2)))

    return theta, mu, sigma


pair_columns = ["sector", "ticker_a", "ticker_b", "alpha", "hedge_ratio",
                "adf_pvalue", "theta", "mu", "sigma", "half_life",
                "stat_std_dev", "start", "end"]

# kalman filter state on the last formation day (alpha, hedge_ratio, the
# symmetric P as P00 P01 P11, R, lPb_mean, sigma_alpha, sigma_beta) followed by
# the ADF test and OU fit of the kalman spread over the filtered window
kalman_columns = ["sector", "ticker_a", "ticker_b", "alpha", "hedge_ratio",
                  "P00", "P01", "P11", "R", "lPb_mean", "sigma_alpha",
                  "sigma_beta", "adf_pvalue", "theta", "mu", "sigma",
                  "half_life", "stat_std_dev", "start", "end"]

def write_pairs(filename, sector_pairs, start, end, columns=pair_columns):
    # writes the cointegrated pairs to a whitespace separated table, one row
    # per pair, with the column names on a commented header line
    # sector_pairs should be {sector: [((ticker_a, ticker_b), params), ...]}
    # where params are the values of columns between ticker_b and start, e.g.
    # for pair_columns params = [alpha, hedge_ratio, adf_pvalue, theta, mu,
    # sigma, half_life, stat_std_dev]
    with open(filename, 'w') as outfile:
        outfile.write('# ' + ' '.join(columns) + '\n')
        for key in sector_pairs:
            for pair, params in sector_pairs[key]:
                t1, t2 = pair
                # float() so 1-element arrays (e.g. sigma from fit_SDE) print as numbers
                values = ' '.join('{:.10g}'.format(float(np.squeeze(p))) for p in params)
                outfile.write('{} {} {} {} {} {}\n'.format(key, t1, t2, values, start, end))


def read_pairs(filename):
    # reads a file written by write_pairs into a numpy structured array,
    # columns are accessed by name, e.g. pairs["ticker_a"], pairs["hedge_ratio"]
    pairs = np.genfromtxt(filename, names=True, dtype=None, encoding=None)
    # a file with a single pair comes back 0-dimensional
    return np.atleast_1d(pairs)

def BFD(y,n,d):
    return (y[1:]-y[:n-1])/d


def pairs_backtest(Pa, Pb, lPa, lPb, z, hedge_ratio, z_in=2., z_out=1.,
                   lag=10, dt=1./252, trend_filter=False):
    # trades the standardized spread z of one pair
    # enter when |z| > z_in, exit when |z| < z_out:
    #   z > z_in  -> short the spread (s = -1): short A, long hedge_ratio of B
    #   z < -z_in -> long the spread  (s = +1): long A, short hedge_ratio of B
    # the signal uses the close of day t and the trade fills at the close of
    # day t+1, so only information available at time t is used
    # sizing is $1 of A and $hedge_ratio of B at entry, the shares are held
    # until exit (no rebalancing), any open position is closed on the last day
    # if trend_filter, only enter when the leg being bought has fallen over
    # the last lag days, exits are never filtered
    # Pa, Pb prices, lPa, lPb log prices, z and hedge_ratio timeseries
    # returns shares_a, shares_b (shares held at the close of each day) and
    # trades, a list of (day, leg, shares, price) with shares > 0 a purchase
    # and shares < 0 a sale
    nt = Pa.size
    shares_a = np.zeros(nt)
    shares_b = np.zeros(nt)
    trades = []

    # current inventory and state (+1 long spread, -1 short spread, 0 flat)
    n_a, n_b = 0., 0.
    state = 0

    for t in range(lag, nt-1):
        if (state == 0):
            if (z[t] > z_in):
                s = -1
            elif (z[t] < -z_in):
                s = 1
            else:
                s = 0
            # no entries that would fill on the last day
            if (t+1 == nt-1):
                s = 0
            if (s != 0 and trend_filter):
                # trend of the leg being bought over the last lag days
                lP_buy = lPa if (s == 1) else lPb
                trend = np.sum(BFD(lP_buy[t-lag:t+1], lag+1, dt))
                if (trend >= 0):
                    s = 0
            if (s != 0):
                n_a = s/Pa[t+1]
                n_b = -s*hedge_ratio[t]/Pb[t+1]
                trades += [(t+1, 'A', n_a, Pa[t+1]), (t+1, 'B', n_b, Pb[t+1])]
                state = s
        elif (np.abs(z[t]) < z_out):
            # exit, sell back what we hold
            trades += [(t+1, 'A', -n_a, Pa[t+1]), (t+1, 'B', -n_b, Pb[t+1])]
            n_a, n_b = 0., 0.
            state = 0
        shares_a[t+1] = n_a
        shares_b[t+1] = n_b

    # close anything still open on the last day
    if (state != 0):
        trades += [(nt-1, 'A', -n_a, Pa[nt-1]), (nt-1, 'B', -n_b, Pb[nt-1])]
        shares_a[nt-1] = 0.
        shares_b[nt-1] = 0.

    return shares_a, shares_b, trades
