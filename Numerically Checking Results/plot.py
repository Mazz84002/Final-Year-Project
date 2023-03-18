import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import quad
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.integrate import solve_bvp
from scipy import constants
from sympy import *

L = 1
mu_0 = constants.mu_0
eps_0 = constants.epsilon_0
omega = 10**9 # 1[GHz]
Z_L = 1e5 # 100k ohms load
# Defining Gamma
z_vals = np.linspace(0, 1, 100) # Generate 1000 equally spaced points between 0 and 0.000001

def Gamma(z):
    return 0.5*np.exp(-1e2*z)
def dGamma_dz(z):
    return -50*np.exp(-1e2*z)
# defining a from hese values
def a(z):
    return -2 * dGamma_dz(z) * 1/(1-(Gamma(z))**2)
def b(z):
    return 4*omega*np.sqrt(mu_0) * (Gamma(z)/(1-(Gamma(z))**2))
# convert a and b to arrays

a_r = [a(z_val) for z_val in z_vals]
b_r = [b(z_val) for z_val in z_vals]
I = [ np.exp(-quad(a, L, z_val)[0]) for z_val in z_vals ]
# We first find I(z)b(z) and we interpolate this function
d = [ (I[i]*b_r[i]) for i in range(0, 100) ]
d_interp = interp1d(np.linspace(0, 1, 100), d, kind='cubic')
D = [ quad(d_interp, L, z_val)[0] for z_val in z_vals ]
eps = [ ( (I[i])/( I[99]*(Z_L/np.sqrt(mu_0)) + 1j*D[i] ) )**2 for i in range(0, 100) ]


def a(z):
    return -2 * dGamma_dz(z) * 1/(1-(Gamma(z))**2)
def b(z):
    return 4*omega*np.sqrt(mu_0) * (Gamma(z)/(1-(Gamma(z))**2))
def d_xi_dz(z, xi):
    return -a(z)*xi-b(z)*xi**2;
req_xi = odeint(d_xi_dz, y0=np.sqrt(eps[0].real), t=z_vals, tfirst=True)
req_eps = np.multiply(req_xi, req_xi)
plt.plot(z_vals, [abs(val)/eps_0 for val in eps], label='my solution')
plt.plot(z_vals, [abs(val)/eps_0 for val in req_eps], label='python ode solver')
plt.legend()
plt.grid()
plt.show()


L = 1
mu_0 = constants.mu_0
omega = 10**9 # 1[GHz]
Z_L = 1e5 # 100k ohms load

# Defining Gamma
def Gamma(t):
    return 0.5*np.exp(-(2*np.pi/omega)*t)
def dGamma_dt(t):
    return -0.5*(2*np.pi/omega)*np.exp(-(2*np.pi/omega)*t)
# defining a from hese values
def a(t):
    return -2 * dGamma_dt(t) * 1/(1-(Gamma(t))**2)
def b(t):
    return 4*omega*np.sqrt(mu_0) * (Gamma(t)/(1-(Gamma(t))**2))

# convert a and b to arrays
t_vals = np.linspace(0, 1, 100)

a_r = [a(t_val) for t_val in t_vals]
b_r = [b(t_val) for t_val in t_vals]
I = [ np.exp(-quad(a, 0, t_val)[0]) for t_val in t_vals ]

# We first find I(z)b(z) and we interpolate this function
d = [ (I[i]*b_r[i]) for i in range(0, 100) ]
d_interp = interp1d(np.linspace(0, 1, 100), d, kind='cubic')
D = [ quad(d_interp, 0, t_val)[0] for t_val in t_vals ]

dGamma_dt_r = [dGamma_dt(t_val) for t_val in t_vals]

xi_0 = (a_r[0]-2*dGamma_dt_r[0])/b_r[0]

eps = [ ( (I[i])/( I[0]*( 1/xi_0  ) + 1j*D[i] ) )**2 for i in range(0, 100) ]


# defining a from hese values
def a(t):
    return -2 * dGamma_dt(t) * 1/(1-(Gamma(t))**2)
def b(t):
    return 4*omega*np.sqrt(mu_0) * (Gamma(t)/(1-(Gamma(t))**2))

def d_xi_dt(t, xi):
    return -a(t)*xi-b(t)*xi**2;

req_xi = odeint(d_xi_dt, y0=xi_0, t=t_vals, tfirst=True)
req_eps = np.multiply(req_xi, req_xi)

plt.plot(t_vals, [abs(val)/eps_0 for val in eps], label='my solution')
plt.plot(t_vals, [abs(val)/eps_0 for val in req_eps], label='python ode solver')
plt.legend()
plt.grid()
plt.show()