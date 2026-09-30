import numpy as np
import buhl as db
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

tstop = 1.
num_dt = 8
step_sizes = np.array([10**-(1+i/2) for i in range(num_dt)])
strong_error = np.zeros_like(step_sizes)

cmap = plt.get_cmap('cool')
colors_i = np.linspace(0, 1, num_dt)
rgba_colors = cmap(colors_i)

g2a = True

fig, ax = plt.subplots(2)

for i, dt in enumerate(step_sizes):
    print('Simulating with dt = {}'.format(dt))
    nt = int(tstop/dt)

    # two timeseries, one for Euler-Maruyama, one for the exact
    # discretization
    x_em = np.zeros(nt+1)
    dW = np.zeros(nt)
    x_exact = np.zeros(nt+1)

    x0 = 0.04
    x_em[0] = x0
    x_exact[0] = x0

    t = np.zeros(nt+1)

    if (g2a): 
        theta = 0.1
        sigma = 0.01
        mu = 0.03
        # to 
        b = np.exp(-theta*dt)
        a = mu*(1.-b)
        var_e = ((sigma**2)/(2*theta))*(1.-b**2)
    else:
        a = 100
        b = 0.8
        var_e = 1
        # to 
        theta = -np.log(b)/dt
        mu = a/(1.-b)
        sigma = np.sqrt(-2.*var_e*np.log(b)/(dt*(1-b**2)))


    if (theta*sigma <= 0):
        print("Warning: computed theta or sigma is negative")

    def deterministic(x,t,dt):
        # used in EM as A in A(x,t,dt)*dt
        return theta*(mu-x)

    def stochastic(x,t,dt):
        return sigma

    
    for j in range(nt):
        x_em[j+1], dW[j] = db.euler_maruyama(x_em[j],t[j],deterministic,stochastic,dt)
        x_exact[j+1] = a + b*x_exact[j] + np.sqrt(var_e)*dW[j]
        t[j+1] = t[j] + dt
   
    ax[0].plot(t,x_em,'-',color=rgba_colors[i],linewidth=0.5)
    ax[0].plot(t,x_exact,'--',color=rgba_colors[i],linewidth=1.0)
    ax[0].set_xlabel(r'$t$')

    #expected_x = mu + (x0 - mu)*np.exp(-theta*t) 
    strong_error[i] = max(np.abs(x_em - x_exact))


norm = mcolors.LogNorm(vmin=step_sizes[-1], vmax=step_sizes[0])
colormap = cm.cool_r
mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
mappable.set_array([]) 
cax = ax[0].inset_axes([1.02, 0., 0.03, 1.])  # just right of ax[0], same height/bottom
cbar = fig.colorbar(mappable, cax=cax)
cbar.set_label(r'$\Delta t$')

ax[1].loglog(step_sizes,strong_error,'bo-')
ax[1].loglog(step_sizes,1e-3*step_sizes**(1),'b--',label=r'$O(\Delta t)$')
ax[1].set_xlabel(r'$\Delta t$')
ax[1].legend()
ax[1].set_title(r'Computed using $E(X_t)$')

fig.tight_layout()
fig.savefig('problem2.png',dpi=500)
