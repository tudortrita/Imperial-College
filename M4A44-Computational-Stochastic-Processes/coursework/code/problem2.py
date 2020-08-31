""" problem2.py - Script containing code used for Problem 2 of the Coursework.
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
import scipy.optimize
import scipy.stats

SAVEFIG = True
SHOWFIG = True

matplotlib.rc('font', size=14)
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', markersize=10)
matplotlib.rc('figure.subplot', hspace=.4)

# Question 2.4 Code and Figures
def ornstein_uhlenbeck_trajectories(dt, M, N, mu, theta, sigma):
    """Computes OU-Trajectories for Process (4)

    Parameters
    ----------
    dt : float
        Time-step parameter in the function.
    M : int
        Number of trajectories to compute.
    N : int
        Size of each trajectories.

    Returns
    -------
    np.ndarray
    """
    X = np.zeros((N+1, M), dtype=float)
    xi = np.random.randn(N, M)
    X[0, :] = 1 + np.random.uniform(low=0, high=1, size=M)  # Initial Condition

    # Precomputing Constants:
    expdt = np.exp(-theta*dt)
    sqrt_term = sigma*np.sqrt(0.5*(1 - np.exp(-2*theta*dt)/theta))

    for i in range(N):
        X[i+1] = mu + expdt*(X[i] - mu) + xi[i]*sqrt_term
    return X

dt = 0.01
mu, theta, sigma = -1, 1, np.sqrt(2)
T = 1
M, N = 10000, 100

X = ornstein_uhlenbeck_trajectories(dt, M, N, mu, theta, sigma)

# Calculating 99% confidence interval for E1:= E[X_T] & E2:= E[X_T^2]
zstar = scipy.stats.norm.ppf(0.995)  #=2.576...

mean_Xt2 = np.mean(X[-1]**2)
var_Xt2 = np.var(X[-1]**2)
std_Xt2 = np.sqrt(var_Xt2)

CI2_low = mean_Xt2 - zstar*std_Xt2/np.sqrt(M)
CI2_high = mean_Xt2 + zstar*std_Xt2/np.sqrt(M)
print(f"The 99% CI for E[X_t^2]: [{CI2_low}, {CI2_high}]")

# Plot of 20 Trajectories
x = np.linspace(0, 1, N+1)
fig = plt.figure()
plt.plot(x, X[:, :20])
plt.xlabel("t")
plt.ylabel(r"$X_t$")
plt.title("Figure 2.4.1 - Plot of 20 Trajectories")
if SAVEFIG:
    plt.savefig("../figures/fig2.4.1.png")
if SHOWFIG:
    plt.show()

# Question 2.6: Numerical Calculation of MLE
def minus_PDF(theta, x, mu, sigma, dt, N):
    std = sigma*np.sqrt((1 - np.exp(-2*theta*dt))/(2*theta))
    sum_term = np.sum(np.abs(x[1:] - (mu + np.exp(-theta*dt)*(x[:-1] - mu)))**2)
    return - np.abs(1/np.sqrt(2*np.pi*std))**N * np.exp(-1/(2*dt)*sum_term)

def minus_logPDF(theta, x, mu, sigma, dt, N):
    std = sigma*np.sqrt((1 - np.exp(-2*theta*dt))/(2*theta))
    sum_term = np.sum(np.abs(x[1:] - (mu + np.exp(-theta*dt)*(x[:-1] - mu)))**2)
    logPDF = N*np.log(np.abs(1/np.sqrt(2*np.pi*std**2))) + -1/(2*std**2)*sum_term
    return - logPDF

theta = 1
dt = 0.1
M, N = 1, 10**7
mu = -1
sigma = np.sqrt(2)
X = ornstein_uhlenbeck_trajectories(dt, M, N, mu, theta, sigma)
theta_mle = scipy.optimize.fminbound(logPDF, 0, 100, args=(X, mu, sigma, dt, N))
print(f"Theta MLE = {theta_mle:.6f} - Real Theta = {theta}")

# Question 2.7: Posterior Distributions
def joint_PDF(theta, x, mu, sigma, dt, N):
    sigma_pdf = np.exp(-(theta-2)**2 / 2) * 1/np.sqrt(2*np.pi)
    std = sigma*np.sqrt((1 - np.exp(-2*theta*dt))/(2*theta))
    sum_term = np.sum(np.abs(x[1:] - (mu + np.exp(-theta*dt)*(x[:-1] - mu)))**2)
    X_PDF = np.abs(1/np.sqrt(2*np.pi*std**2))**N * np.exp(-1/(2*std**2)*sum_term)
    return sigma_pdf*X_PDF

def log_joint_PDF(theta, x, mu, sigma, dt, N):
    log_sigma_pdf = np.log(1/np.sqrt(2*np.pi)) - (theta-2)**2 / 2
    X_log_PDF = - minus_logPDF(theta, x, mu, sigma, dt, N)
    return log_sigma_pdf + X_log_PDF

def minus_joint_PDF(theta, x, mu, sigma, dt, N):
    return - joint_PDF(theta, x, mu, sigma, dt, N)

theta_array = np.linspace(-10, 10)
pdf_val = []
for theta in theta_array:
    pdf_val.append(np.exp(log_joint_PDF(theta, X, mu, sigma, dt, N)))

plt.figure()
plt.plot(theta_array, pdf_val)
plt.xlabel(r"$\theta$")
plt.ylabel("Conditional PDF")
plt.title(r"Figure 2.7.1 - Conditional Distribution of $\theta$ given previous observations")
if SAVEFIG:
    plt.savefig("../figures/fig2.7.1.png")
if SHOWFIG:
    plt.show()
