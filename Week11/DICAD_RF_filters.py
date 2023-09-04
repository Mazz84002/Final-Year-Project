# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import constants
import itertools
import skrf as rf
import pandas as pd

plt.rcParams['lines.linewidth'] = 0.8

# %%
fstart = -3
fstop = 40
fpoints = 100

f = np.logspace(fstart, fstop, fpoints)

# %%
L = 1e-7
Cm1 = 3e-9
Cm2 = 3e-10
Ca = 3e-20

# %%
num_of_sections = 8
num_of_combinations = 2**num_of_sections

# Generate all possible combinations of an 8-bit input
input_combinations = list(itertools.product([0, 1], repeat=8))

# Convert the combinations to a NumPy array
input_matrix = np.array(input_combinations)

Cm = input_matrix*Cm1 + (1-input_matrix)*Cm2 # capacitance values for every single state

# %%
# Assume negligible looses
R = 1e-4
Gm = 1e-4
Ga = 1e-6

l = 1e-4 / num_of_sections # length of each line

# %%
def invert_mat(A):
    inv_A = np.zeros((len(A[0, :, 0]), len(A[:, 0, 0]), len(A[0, 0, :])), dtype=complex)
    for i in range(len(A[0, 0])):
        inv_A[:, :, i] = np.linalg.inv(A[:, :, i])

    return inv_A

def multiply_mat(A, B):
    C = np.zeros((len(A[0, :, 0]), len(A[:, 0, 0]), len(A[0, 0, :])), dtype=complex)
    for i in range(len(A[0, 0, :])):
        C[:, :, i] = A[:, :, i] @ B[:, :, i]

    return C

# %%
# Define odd and even mode parameters

def odd_mode_params(f, L, Cm, Ca, Gm, Ga):
    w = 2*np.pi*f
    Co = 2*Cm - Ca
    Go = 2*Gm + Ga
    Lo = L*(1-1e-3)
    Ro = R
    gamma_o = 1j*(2*np.pi*np.sqrt(Lo*Co))
    Z0o = np.sqrt((Ro+1j*w*Lo)/(Go + 1j*w*Co))

    f0_o = 1/(2*np.pi*np.sqrt(Lo/Co)) # frequency at which the coupled lines are a quarter-wavelength long electrically when excited in the odd mode
    theta_o = (np.pi/2)*(f/f0_o)

    return Lo, Co, Go, Ro, gamma_o, Z0o, theta_o

def even_mode_params(f, L, Cm, Ca, Gm, Ga):
    w = 2*np.pi*f
    Ce = Ca
    Ge = Ga
    Le = L*(1+1e-3)
    Re = R
    gamma_e = 1j*(2*np.pi*w*np.sqrt(Le*Ce))
    Z0e = np.sqrt((Re+1j*w*Le)/(Ge + 1j*w*Ce))

    f0_e = 1/(2*np.pi*np.sqrt(Le/Ce)) #frequency at which the coupled lines are a quarter-wavelength long electrically when excited in the even mode,
    theta_e = (np.pi/2)*(f/f0_e)

    return Le, Ce, Ge, Re, gamma_e, Z0e, theta_e

def create_Z_matrix(Z0e, cot_theta_e, csc_theta_e, Z0o, cot_theta_o, csc_theta_o):

    Z11 = -(1j/2) * (Z0e*cot_theta_e + Z0o*cot_theta_o)
    Z12 = -(1j/2) * (Z0e*cot_theta_e - Z0o*cot_theta_o)
    Z13 = -(1j/2) * (Z0e*csc_theta_e - Z0o*csc_theta_o)
    Z14 = -(1j/2) * (Z0e*csc_theta_e + Z0o*csc_theta_o)

    Z = np.zeros((4, 4, len(Z11)), dtype=complex)

    Z[0][0][:] = Z11
    Z[1][1][:] = Z[0][0][:]
    Z[2][2][:] = Z[0][0][:]
    Z[3][3][:] = Z[0][0][:]

    Z[0][1][:] = Z12
    Z[1][0][:] = Z[0][1][:]
    Z[2][3][:] = Z[0][1][:]
    Z[3][2][:] = Z[0][1][:]

    Z[0][2][:] = Z13
    Z[2][0][:] = Z[0][2][:]
    Z[1][3][:] = Z[0][2][:]
    Z[3][1][:] = Z[0][2][:]

    Z[0][3][:] = Z14
    Z[3][0][:] = Z[0][3][:]
    Z[1][2][:] = Z[0][3][:]
    Z[2][1][:] = Z[0][3][:]

    return Z

def find_Zin(ZL, Zc, beta, l):
    return Zc * ( (ZL + 1j*Zc*np.tan(beta*l))/(Zc + 1j*ZL*np.tan(beta*l)) )

def find_imput_impedances(num_of_sections, index, L, C_curr, l, f):
    if (index == num_of_sections-1):
        Z03 = 50*np.ones(len(f))
    else:
        Z03 = 50*np.ones(len(f))
        while(index != num_of_sections-1):
            Z03 = find_Zin(Z03, np.sqrt(L/C_curr[index]), 2*np.pi*np.sqrt(L*C_curr[index]), l)
            index += 1

    if (index == 0):
        Z01 = 50*np.ones(len(f))
    else:
        temp = 0
        Z01 = 50*np.ones(len(f))
        while(temp != index-1):
            Z01 = find_Zin(Z03, np.sqrt(L/C_curr[index]), 2*np.pi*np.sqrt(L*C_curr[index]), l)
            temp += 1
    
    Z02 = Z01
    Z04 = Z03
    return Z01, Z02, Z03, Z04

def create_F(Z01, Z02, Z03, Z04):
    F = np.zeros((4, 4, len(Z01)), dtype=complex)
    F[0][0][:] = 1/(2*np.sqrt(Z01))
    F[1][1][:] = 1/(2*np.sqrt(Z02))
    F[2][2][:] = 1/(2*np.sqrt(Z03))
    F[3][3][:] = 1/(2*np.sqrt(Z04))
    return F

def create_G(Z01, Z02, Z03, Z04):
    G = np.zeros((4, 4, len(Z01)), dtype=complex)
    G[0][0][:] = Z01
    G[1][1][:] = Z02
    G[2][2][:] = Z03
    G[3][3][:] = Z04
    return G

def z2s(Z, F, G):
    # Calculate Z - G* and Z + G
    Z_minus_G_star = Z - np.conj(G)
    Z_plus_G = Z + G
    
    # Calculate the inverse of Z + G
    Z_plus_G_inv = invert_mat(Z_plus_G)
    
    # Calculate F^(-1)
    F_inv = invert_mat(F)
    
    # Calculate S = F(Z - G*)(Z + G)^(-1)F^(-1)
    S = multiply_mat(multiply_mat(F, Z_minus_G_star), multiply_mat(Z_plus_G_inv, F_inv))
    
    return S

def s2z(S, F, G):
    # Calculate the identity matrix I of the same shape as S
    I = np.zeros(S.shape, dtype=complex)
    for i in range(S.shape[2]):
        I[:, :, i] = np.eye(4, dtype=complex)
    
    # Calculate (I - S)^(-1)
    I_minus_S_inv = invert_mat(I - S)
    
    # Calculate SG + G*
    SG_plus_G_star = multiply_mat(S, G) + np.conj(G)
    
    # Calculate F^(-1)
    F_inv = invert_mat(F)
    
    # Calculate Z = F^(-1)(I - S)^(-1)(SG + G*)F
    Z = multiply_mat(multiply_mat(F_inv, I_minus_S_inv), multiply_mat(SG_plus_G_star, F))
    
    return Z

# %%
# initialise S parameters matrix

S = np.zeros((4, 4, len(f)), dtype=complex)

# %%
for j in range(num_of_sections-1, -1, -1):

    Lo, Co, Go, Ro, gamma_o, Z0o, theta_o = odd_mode_params(f, L, Cm[0][j], Ca, Gm, Ga)
    Le, Ce, Ge, Re, gamma_e, Z0e, theta_e = even_mode_params(f, L, Cm[0][j], Ca, Gm, Ga)

    Y0e = 1/(Z0e)
    Y0o = 1/(Z0o)

    cot_theta_o = 1/np.tan(theta_o)
    cot_theta_e = 1/np.tan(theta_e)

    csc_theta_o = 1/np.sin(theta_o)
    csc_theta_e = 1/np.sin(theta_e)

    Z = create_Z_matrix(Z0e, cot_theta_e, csc_theta_e, Z0o, cot_theta_o, csc_theta_o)
    Z01, Z02, Z03, Z04 = find_imput_impedances(num_of_sections, j, L, Cm[0][:], l, f)
    F = create_F(Z01, Z02, Z03, Z04)
    G = create_G(Z01, Z02, Z03, Z04)


    S_curr = z2s(Z, F, G) # convert to S parameters for cascading
    if (j == 0):
        S = S_curr
    else:
        S = multiply_mat(S_curr, S)

# update F and G
F = create_F(50*np.ones(len(f)), 50*np.ones(len(f)), 50*np.ones(len(f)), 50*np.ones(len(f)))
G = create_G(50*np.ones(len(f)), 50*np.ones(len(f)), 50*np.ones(len(f)), 50*np.ones(len(f)))

Z = s2z(S, F, G)

# %%
def create_reduced_F(f):
    F = np.zeros((2, 2, len(f)), dtype=complex)
    F[0][0][:] = 1/(2*np.sqrt(50))
    F[1][1][:] = 1/(2*np.sqrt(50))

    return F

def create_reduced_G(f):
    G = np.zeros((2, 2, len(f)), dtype=complex)
    G[0][0][:] = 50
    G[1][1][:] = 50

    return G

# %% [markdown]
# # Low Pass Filters

# %%
# V2 = 0 and I3 = 0 - reduces the matrix to that of a 2 port system

def reduce_low_pass(Z):
    return Z[np.ix_([0, 3], [0, 3])]

Z_lpf = reduce_low_pass(Z)

S_lpf = z2s(Z_lpf, create_reduced_F(f), create_reduced_G(f))

# %% [markdown]
# # Band Pass Filters

# %%
# I2 = 0 and I4 = 0 - reduces the matrix to that of a 2 port system

def reduce_band_pass(Z):
    return Z[np.ix_([0, 2], [0, 2])]

Z_bpf = reduce_band_pass(Z)

S_bpf = z2s(Z_bpf, create_reduced_F(f), create_reduced_G(f))


