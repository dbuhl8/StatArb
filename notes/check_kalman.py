import os, sys, numpy as np, yfinance as yf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import buhl as db

def raw_reference(lPa, lPb, Q):
    # Kalman filter written directly in the uncentered variables
    nt = lPa.size; fnt = int(nt/4)
    A = np.stack((np.ones(fnt), lPb[:fnt]), axis=1)
    s, ssr, _, _ = np.linalg.lstsq(A, lPa[:fnt])
    R = ssr[0]/(fnt-2); P = R*np.linalg.inv(A.T@A)
    al = np.full(nt, s[0]); be = np.full(nt, s[1]); sp = np.zeros(nt)
    sp[:fnt] = lPa[:fnt] - s[0] - s[1]*lPb[:fnt]
    for i in range(fnt, nt):
        H = np.array([1., lPb[i]]); P = P + Q
        e = lPa[i] - al[i-1] - be[i-1]*lPb[i]
        K = P@H/(H@P@H + R)
        al[i] = al[i-1] + K[0]*e; be[i] = be[i-1] + K[1]*e
        P = (np.eye(2) - np.outer(K, H))@P; sp[i] = e
    return al, be, sp, P

def half_life(s):
    m = np.polyfit(s[:-1], s[1:], 1)[0]
    return np.log(2)/-np.log(m) if 0 < m < 1 else np.inf

dt = 1/252
px = yf.download(["SSD","LIN"], start="2015-01-01", end="2023-12-31", auto_adjust=True, progress=False)["Close"].dropna()
lPa, lPb = np.log(px["SSD"].values), np.log(px["LIN"].values)
nt = lPa.size; fnt = int(nt/4); xbar = lPb[:fnt].mean()

print("== 1. fit_spread: raw vs centered lPb (initial window)")
S1, b1, (a1, r1, p1), _ = db.fit_spread(lPa[:fnt], lPb[:fnt])
S2, b2, (a2, r2, p2), _ = db.fit_spread(lPa[:fnt], lPb[:fnt] - xbar)
print("max|spread diff| = %.2e, beta diff = %.2e, SSR diff = %.2e, ADF p diff = %.2e"
      % (np.abs(S1-S2).max(), abs(b1-b2), abs(r1-r2), abs(p1-p2)))
print("alpha raw = %.6f, centered alpha - beta*mean = %.6f" % (a1, a2 - b2*xbar))
A = np.stack((np.ones(fnt), lPb[:fnt]), axis=1); Ac = np.stack((np.ones(fnt), lPb[:fnt]-xbar), axis=1)
R = r1/(fnt-2); Praw = R*np.linalg.inv(A.T@A); Pc = R*np.linalg.inv(Ac.T@Ac)
print("corr(alpha,beta) in P0: raw = %.5f, centered = %.5f" % (Praw[0,1]/np.sqrt(Praw[0,0]*Praw[1,1]), Pc[0,1]/np.sqrt(Pc[0,0]*Pc[1,1])))
print("cond(P0): raw = %.2e, centered = %.2e" % (np.linalg.cond(Praw), np.linalg.cond(Pc)))

T = np.array([[1., -xbar], [0., 1.]])
for sa, sb in [(0., 0.), (0.03, 0.01), (0.1, 0.1)]:
    al, be, sp, P = db.kalman_spread(lPa, lPb, dt, sa, sb)
    Qc = np.diag([sa**2, sb**2])*dt
    ral, rbe, rsp, rP = raw_reference(lPa, lPb, T@Qc@T.T)
    print("== 2. centered filter vs raw-variable filter, same model (sa=%g, sb=%g)" % (sa, sb))
    print("   max diff: alpha %.2e  beta %.2e  spread %.2e  P %.2e   identity check |lPa-a-b*lPb-spread| on filtered part: %.2e"
          % (np.abs(al-ral).max(), np.abs(be-rbe).max(), np.abs(sp-rsp).max(), np.abs(P-rP).max(),
             np.abs(lPa[fnt:] - np.r_[al[fnt-1], al[fnt:-1]] - np.r_[be[fnt-1], be[fnt:-1]]*lPb[fnt:] - sp[fnt:]).max()))

print("== 3. same sigmas but diagonal Q in RAW variables (old approach) vs centered")
for sa, sb in [(0.03, 0.01)]:
    _, be_c, sp_c, _ = db.kalman_spread(lPa, lPb, dt, sa, sb)
    _, be_r, sp_r, _ = raw_reference(lPa, lPb, np.diag([sa**2, sb**2])*dt)
    print("   centered: half-life %.1f d, std %.4f, beta %.3f..%.3f" % (half_life(sp_c[fnt:]), sp_c[fnt:].std(), be_c[fnt:].min(), be_c[fnt:].max()))
    print("   raw diag: half-life %.1f d, std %.4f, beta %.3f..%.3f" % (half_life(sp_r[fnt:]), sp_r[fnt:].std(), be_r[fnt:].min(), be_r[fnt:].max()))

print("== 4. SSD/LIN static vs Kalman (defaults sa=0.03, sb=0.01)")
S, b, (a, r, p), _ = db.fit_spread(lPa, lPb)
al, be, sp, P = db.kalman_spread(lPa, lPb, dt)
print("   static: beta %.3f, half-life %.1f d, std %.4f" % (b, half_life(S), S.std()))
print("   kalman: beta %.3f..%.3f, half-life %.1f d, std %.4f (filtered part)" % (be[fnt:].min(), be[fnt:].max(), half_life(sp[fnt:]), sp[fnt:].std()))

print("== 5. synthetic: constant beta = 1.3, alpha = -0.5")
rng = np.random.default_rng(1)
n = 2000; lb = 4 + np.cumsum(0.01*rng.standard_normal(n))
la = -0.5 + 1.3*lb + 0.02*rng.standard_normal(n)
al, be, sp, P = db.kalman_spread(la, lb, dt, 0.0, 0.0)
print("   final alpha %.4f, beta %.4f, beta sd %.4f" % (al[-1], be[-1], np.sqrt(P[1,1])))
al, be, sp, P = db.kalman_spread(la, lb, dt)
print("   defaults: final alpha %.4f, beta %.4f" % (al[-1], be[-1]))
