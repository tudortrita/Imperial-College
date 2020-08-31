""" problem1.py - Script containing code used for Problem 1 of the Coursework.
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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

SAVEFIG = True
SHOWFIG = True

matplotlib.rc('font', size=14)
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', markersize=10)
matplotlib.rc('figure.subplot', hspace=.4)

# Question 1.3 (Region of Stability for different values of theta)
thetas = np.array([0, .25, .5, .75, 1])
x = np.linspace(-5, 1, 1000)

fig, ax = plt.subplots()
for theta in np.flip(thetas):
    sqrt_term = np.maximum(0, 1 - 2*(2*x + x**2 * (1 - 2*theta)))
    y = np.maximum(0, -1 + sqrt_term)
    ax.plot(x, y, label=rf"$\theta$-Milstein,  $\theta = {theta}$")
    ax.fill_between(x, 0*y, y, alpha=.2)

ax.set_title(r"Figure 1.3.1 - Region of mean-square stability of GBM")
ax.set_ylabel(r"$\sigma^2 \, \Delta t$")
ax.set_xlabel(r"$\mu \, \Delta t$")
ax.set_xlim(-3, 1)
ax.set_ylim(0, 5)
ax.grid()
ax.legend(loc="upper center")
if SAVEFIG:
    plt.savefig("../figures/fig1.3.1.png")
if SHOWFIG:
    plt.show()

# Question 1.4 A-Stability
x = np.linspace(-2.5, 1, 10000)
y = np.maximum(0, np.sqrt(-2*x))
y = np.nan_to_num(y, 0)

fig, ax = plt.subplots()
ax.plot(x, y, label="Exact Solution")
ax.fill_between(x, 0*x, y, alpha=.2)
sqrt_term = np.maximum(0, 1 - 2*(2*x + x**2 * (1 - 2*1)))
y = np.maximum(0, -1 + sqrt_term)
ax.plot(x, y, label=r"Best $\theta$-Milstein ($\theta = 1$)")
ax.fill_between(x, 0*x, y, alpha=.2)
ax.set_title("Figure 1.4.1 - Region of mean-square stability of GBM")
ax.set_xlim(-1, 0.5)
ax.set_ylim(0, 2)
ax.legend()

if SAVEFIG:
    plt.savefig("../figures/fig1.4.1.png")
if SHOWFIG:
    plt.show()


# Question 1.5:
def geom_brownian_motion_integrator(t, mu, sigma, theta, m, x0):
    """Calculates numerical solution to Geometric Brownian Motion SDE.

    Parameters
    ----------
    t : np.ndarray
        Array containing times at which the paths are computed.
    mu : float
        Drift coefficient in the SDE.
    sigma : float
        Diffusion coefficient in the SDE.
    theta : float
        Parameter in theta milstein scheme governing 'implicitness'.
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
    def step_theta_milstein(x, dt):
        dw = np.sqrt(dt) * np.random.randn(len(x))
        numerator = (1 + mu*dt*(1 - theta) + sigma*dw + (sigma**2)*((dw**2) - dt)/2)
        denominator = 1 - mu*theta*dt
        return (numerator / denominator * x, dw)

    n_times = len(t)

    # Matrix to store the solutions
    result = np.zeros((n_times, m), dtype=np.float64)
    result[0, :] = x0 * np.ones(m, dtype=np.float64)

    # Matris to store the Brownian motions for computation of exact solution.
    w = np.zeros((n_times, m), dtype=np.float64)

    # Main Loop:
    for i in range(n_times - 1):
        result[i + 1], dw = step_theta_milstein(result[i], t[i+1] - t[i])
        w[i + 1] = w[i] + dw
    return (result, w)


# Code used
dt_star = 1
dt1 = 2*dt_star
dt2 = dt_star / 2
theta = 0.25
mu = -1
sigma = 1
t1 = np.linspace(0, 100, 101)*dt1
t2 = np.linspace(0, 100, 101)*dt2
x0 = 1

sol1, w1 = geom_brownian_motion_integrator(t1, mu, sigma, theta, 100000, x0)
sol2, w2 = geom_brownian_motion_integrator(t2, mu, sigma, theta, 100000, x0)

# Computing estimates
eX_1 = np.mean(np.abs(sol1[-1, :])**2)
eX_2 = np.mean(np.abs(sol2[-1, :])**2)

print(f"Estimate for dt=2: {eX_1}")
print(f"Estimate for dt=0.5: {eX_2}")
