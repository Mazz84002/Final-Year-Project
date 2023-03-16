import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def Gamma(z):
    return np.exp(-z)

def a(z, w, mu0):
    return -2*np.exp(-z)/(1-Gamma(z)**2)

def b(z, w, mu0):
    return 4*w*np.sqrt(mu0)*Gamma(z)/(1-Gamma(z)**2)

def integrand(z, w, mu0):
    return np.exp(-quad(a, z, np.inf, args=(w, mu0))[0])

def I(z, w, mu0):
    return np.exp(-quad(a, z, np.inf, args=(w, mu0))[0])

def y(z, L, w, mu0, ZL):
    integrand_L_z = quad(lambda x: I(x, w, mu0)*b(x, w, mu0), L, z)[0]
    I_L = I(L, w, mu0)
    I_inv_z = 1/I(z, w, mu0)
    I_inv_integrand_L_z = 1/(I_L*ZL*I_inv_z + I_inv_z*integrand_L_z)
    return I_inv_integrand_L_z**2

# Set the parameters
w = 1
mu0 = 1
ZL = 1

# Define the range of z values to plot
z_vals = np.linspace(0, 1, 100)

# Evaluate y(z) for different values of L
L_vals = [1]

# Plot y(z) for each value of L
for L in L_vals:
    y_vals = [y(z, L, w, mu0, ZL) for z in z_vals]
    plt.plot(z_vals, y_vals, label=f'L={L}')
    
# Add labels and legend
plt.xlabel('z')
plt.ylabel('y(z)')
plt.legend()
plt.show()
