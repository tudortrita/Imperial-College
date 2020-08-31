""" Function for part 3"""

import matplotlib.pyplot as plt
import numpy as np

t =np.linspace(0,1.2, 10000)

c1 = (1 - 0.7*t)**2
c2 = (1 - 0.3*t)**2
c3 = (1 - t)

plt.figure(figsize=(7, 5))
plt.xlabel(r'$P_\infty$')
plt.plot(t, c1, 'b', label='$p > p_c$')
plt.plot(t, c2, 'r--', label='$p < p_c$')
plt.plot(t, c3, 'y', label=r'$(1-P_\infty)$')
plt.legend()
plt.title('Plot of ' +
          r'$ (1 - p P_\infty)^q$')
plt.grid()

plt.savefig('fig2.png')
plt.show()
