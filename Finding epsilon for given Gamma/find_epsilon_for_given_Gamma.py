import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as c

L = 1.0
ZL = 1.0

def I(z):
    return np.sin(z**2)

def b(x):
    return 1.0

def epsilon(z):
    numerator = 1.0
    denominator = I(L) * (ZL / np.sqrt(c.mu_0)) * (1 / I(z)) + (1 / I(z)) * np.trapz(I(x) * b(x), dx=0.01, x=np.linspace(L, z, 100))
    return numerator / denominator**2

z = np.linspace(0.1, 10, 100)
eps = epsilon(z)

plt.plot(z, eps)
plt.xlabel('z')
plt.ylabel('$\epsilon(z)$')
plt.show()
