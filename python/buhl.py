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
    return S, hedge_ratio, [alpha, resid,pvalue], result

def kalman_spread(lPa, lPb, dt):
    # fits a kalman filter to a potential spread
    # starts by isolating a quarter of the initial timeseries, fits a
    # least-squares regression on it, and then loops forward and applies the # kalman filter to each following timestep
    # returns the spread and hedge ratio as a timeseries
    nt = np.size(lPa)
    fnt = int(nt/4)
    # initial fit on reduced timeseries
    S, hedge_ratio, other_stuff, pvalue, result = fit_spread(lPa[:fnt], lPb[:fnt])
    alpha, residuals, pvalue = other_stuff
    alpha = np.ones_like(lPa)*alpha
    hedge_ratio = np.ones_like(lPa)*hedge_ratio
    
    # initializing kalman recursion
    P = np.eye(2)
    P[0,0] = pvalue
    P[1,1] = residuals/fnt
    R = 0 # placeholder until R is implemented

    # propagating kalman filter forward
    for i in range(fnt,nt):
        H = [[1], [lPb[i+1]]]
        # compute the error in the prediction
        err = lPa[i+1] - alpha[i] - hedge_ratio[i]*lPb[i+1]
        # propagate the uncertainty forward
        P = P + np.eye(2)*dt
        # compute uncertainty of observations
        OE = np.matmul(H.T, np.matmul(P,H)) + R
        # Kalman Gain
        K = np.matmul(P,H)/OE
        # Update alpha, hedge_ratio, uncertainty
        alpha[i+1] = alpha[i] + K[0]*err
        hedge_ratio[i+1] = hedge_ratio[i] + K[0]*err
        P = np.matmul(np.eye(2) - np.outer(K,H),P)
    return [alpha, hedge_ratio, P]
         
    
def fit_SDE(spread, nt, dt):
    # given a stationary, mean-reverting timeseries
    A = np.stack((spread[:nt-1],np.ones_like(spread[:nt-1])), axis=1)
    coeff, res, _, _ = np.linalg.lstsq(A,spread[1:])
    m, b = coeff

    # Derive properties of the SDE from the interpolation
    theta = -np.log(m)/dt
    mu = b/(1. - m)
    sigma = np.sqrt(res/nt)

    return theta, mu, sigma


pair_columns = ["sector", "ticker_a", "ticker_b", "alpha", "hedge_ratio",
                "adf_pvalue", "theta", "mu", "sigma", "half_life", "start", "end"]

def write_pairs(filename, sector_pairs, start, end):
    # writes the cointegrated pairs to a whitespace separated table, one row
    # per pair, with the column names on a commented header line
    # sector_pairs should be {sector: [((ticker_a, ticker_b), params), ...]}
    # where params = [alpha, hedge_ratio, adf_pvalue, theta, mu, sigma, half_life]
    with open(filename, 'w') as outfile:
        outfile.write('# ' + ' '.join(pair_columns) + '\n')
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


