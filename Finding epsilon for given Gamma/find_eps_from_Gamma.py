import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy import constants
from scipy.integrate import odeint

# plot xi(z) using Numeric python ODEs

L, ZL, w, k = 1, 1, 0.2, 2/3

def Gamma(z):
    return np.exp(-10*z)

def a(z):
    return -2 * derivative(Gamma(z), z, dx=1e-6)

def b(z):
    return 4*1j*w*sqrt(constants.mu_0) * Gamma(z) * 1/(1-(Gamma(z))**2)

def dxidt(xi, z):
    return -a(z)*xi - b(z)*xi^2
xi_L = constants.mu_0/ZL

z = np.linspace(0, L, 1000)
sol = odeint(dxidt, xi_L, z)

xi_sol = sol.T[0]

plt.plot(z, xi_sol)
plt.show()