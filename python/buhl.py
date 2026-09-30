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
    coeff,_,_,_ = np.linalg.lstsq(A,lPa)
    hedge_ratio, constant = coeff
    S = lPa - hedge_ratio*lPb - constant
    # Augmented Dickey-Fuller test to see if the spread is stationary
    adf_result = adfuller(S, result_object=True) # c.f. statsmodels.tsa.stattools.adfuller
    pvalue = adf_result[1]
    # need to extract the ADF result
    if pvalue >= 0.05: 
        #print('Warning: the spread does not pass the ADF test (p = {})'.format(pvalue))
        result = False
    else:
        result = True
    return S, hedge_ratio, result

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


