""" Code for Coursework 2 M3A29"""
import numpy as np
import matplotlib.pyplot as plt

def process_parth(params, eq, times):
    """ Function to simulate process"""
    u, v, N, t, x = params
    ave_array = [x - eq]
    x = np.ones(times)*x
    for k in range(t):
        # Outcomes as in eq.6:
        o1, o2, o3 = x + 4 / N, x - 4 / N, x
        # Probabilities as in eq.6
        p1 = (1 - x)**2 * (1 - u) * v / 2
        p2 = x * (x - 2 / N) * (1 - v) * u / 2
        p3 = 1 - p2 - p1
        # Calc. x(t + 1) for each realisation
        for i in range(times):
            x[i] = np.random.choice([o1[i], o2[i], o3[i]],
                                    p=[p1[i], p2[i], p3[i]])
        x_ave = np.mean(x - eq)  # Calculating expression
        ave_array.append(x_ave)
    return ave_array

N_array = [10000, 15000, 20000, 25000]
colours = ['b', 'r--', 'y', 'm--']
u, v, t = 0.5, 0.25, 100000
alpha, beta = 2 * (1 - u) * v, 2 * (1 - v) * u
x_0, times = 0.5, 200
times = 200  # No. of realisations
formula = 1 / (1 + np.sqrt(beta / alpha))

plt.figure(figsize=(12, 9))
for c, N in enumerate(N_array):
    params = u, v, N, t, x_0
    change_array = process_parth(params, formula, times)
    plt.plot(range(t + 1), change_array, colours[c], label=('$N = $' + str(N)))

plt.xlabel('t')
plt.ylabel(r'$ <x(t)> - (1 + \sqrt{ \beta / \alpha})^{-1}$')
plt.title('Plot of formula:' +
          r'$ <x(t)> - (1 + \sqrt{ \beta / \alpha})^{-1}$')
plt.grid()
plt.legend(loc='upper right')
plt.savefig('fig1.png')
plt.show()
