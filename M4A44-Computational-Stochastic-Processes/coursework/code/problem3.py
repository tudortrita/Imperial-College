""" problem3.py - Script containing code used for Problem 3 of the Coursework.
Tudor Trita Trita
CID: 01199397
MSci Mathematics

M4A44 - Computational Stochastic Processes.
"""
try:
    from os import chdir
    chdir("code")
except:
    pass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SAVEFIG = True
SHOWFIG = True

matplotlib.rc('font', size=14)
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', markersize=10)
matplotlib.rc('figure.subplot', hspace=.4)

# Question 3.6 Code:
def hierarchical_check(a):
    try:
        a = np.array(a)
    except:
        pass
    res = np.sum(np.ones(a.size)*2) - np.sum(a)
    return res


def geom_brownian_motion_integrator(t, mu, sigma, m, x0):
    """Calculates numerical solution to Geometric Brownian Motion Stratonovich SDE.

    Parameters
    ----------
    t : np.ndarray
        Array containing times at which the paths are computed.
    mu : float
        Drift coefficient in the SDE.
    sigma : float
        Diffusion coefficient in the SDE.
    m : int
        Number of paths to compute.
    x0 : float
        Value for the initial condition.

    Returns
    -------
    tuple (np.ndarray, np.ndarray)
        Tuple containing two Numpy ndarrays for the computed numerical solution
        in the first element and Brownian Motion Paths in the second element.
    """
    n_times = len(t)

    # Precomputing Constants
    c3 = (mu**2)/2
    c4 = (sigma**2)/2
    c5 = mu*sigma
    c6 = mu*(sigma**2) / 2
    c7 = (sigma**3)/6
    c8 = (sigma**4)/24

    # Matrix to store the solutions
    result = np.zeros((n_times, m), dtype=np.float64)
    result[0, :] = x0 * np.ones(m, dtype=np.float64)

    # MatriX to store the Brownian motions for computation of exact solution.
    w = np.zeros((n_times, m), dtype=np.float64)

    # Main Loop:
    for i in range(n_times - 1):
        dt = t[i+1] - t[i]
        dw = np.sqrt(dt) * np.random.randn(m)

        sum_term = (mu*dt + sigma*dw + c3*(dt**2) + c4*(dw**2) + c5*dt*dw
                  + c6*dt*(dw**2) + c7*(dw**3) + c8*(dw**4))

        result[i + 1] = result[i]*(1 + sum_term)
        w[i + 1] = w[i] + dw
    return (result, w)

def geom_brownian_motion_exact(t, w, mu, sigma, x0):
    """Calculates exact solution to Geometric Brownian Motion Stratonovich SDE.

    Parameters
    ----------
    t : np.ndarray
        Array containing times at which the paths are computed.
    w : type
        Array containing brownian motions.
    x0 : type
        Initial condition.

    Returns
    -------
    np.ndarray
        Array containing exact solution.
    """
    t.shape = (len(t), 1)
    return x0 * np.exp(mu*t + sigma * w)

# Code to check that the Implementation is correct and works well:
x0 = 1
mu = -1
sigma = 1

dt = 0.005
invdt = int(1/dt)

t = np.linspace(0, invdt, invdt+1)*dt
m = 2

X, w = geom_brownian_motion_integrator(t, mu, sigma, m, x0)
Xexact = geom_brownian_motion_exact(t, w, mu, sigma, x0)

fig = plt.figure()
plt.plot(t, X, label="Numerical Scheme")
plt.plot(t, Xexact, "--", label="Exact")
plt.title("Figure 3.6.1 - Two Paths of Stratonovich GBM SDE")
plt.legend()
plt.xlabel("t")
plt.ylabel(r"$X_t$")
if SAVEFIG:
    plt.savefig("../figures/fig3.6.1.png")
if SHOWFIG:
    plt.show()


# Showing Order 2 Strong Convergence:
def plot_errors(dts, errors, error_type, method, SAVEFIG=False, SHOWFIG=True):
    # Fit to estimate order of convergence
    coeffs = np.polyfit(np.log2(dts), np.log2(errors), 1)

    # Plot
    fig, ax = plt.subplots()
    ax.set_title("{} error of the {} scheme".format(error_type, method))
    if error_type == "Strong":
        ylabel = r"$\sup \{ |X^{\Delta t}_n  - X_{n \Delta t}|:" \
                + "n \Delta t \in [0, T] \}$"
    elif error_type == "Weak":
        ylabel = r"$|E[f(X^{\Delta t}_{T/\Delta t}] - E[f(X_t)] |$"
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(ylabel)
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=2)
    ax.plot(dts, errors, linestyle='', marker='.')
    ax.plot(dts, 2**coeffs[1] * (dts)**coeffs[0],
            label=r'${:.2f} \, \times \, \Delta t^{{ {:.2f} }}$'.
            format(2**coeffs[1], coeffs[0]))
    ax.legend()
    if SAVEFIG:
        plt.savefig(f"../figures/{SAVEFIG}.png")
    if SHOWFIG:
        plt.show()

def strong_error(x, x_exact):
    sup_interval = np.max(np.abs(x - x_exact), axis=0)
    return np.mean(sup_interval)

T, n = 1, 100
m, len_ns = 300, 10
ns = np.logspace(2, 4, len_ns)
ns = np.array([int(n) for n in ns])
strong_errors = np.zeros(len_ns)

# Calculating Strong Order:
for i, n in enumerate(ns):
    t = np.linspace(0, T, n)
    x, w = geom_brownian_motion_integrator(t, mu, sigma, m, x0)
    x_exact = geom_brownian_motion_exact(t, w, mu, sigma, x0)
    strong_errors[i] = strong_error(x, x_exact)
plot_errors(T/ns, strong_errors, "Strong", "Stratonovich SDE Iteration", SAVEFIG="fig3.6.2")
