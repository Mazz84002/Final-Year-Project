import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import constants
import sympy
import random
import pandas as pd
import itertools


num_of_sections = 8
num_of_combinations = 2**num_of_sections
l = 1000
f = np.logspace(np.log10(1e4), np.log10(1e15), l)

L0 = 1e-7

C1 = 3e-9
C2 = 3e-10


# Generate all possible combinations of an 8-bit input
input_combinations = list(itertools.product([0, 1], repeat=8))

# Convert the combinations to a NumPy array
input_matrix = np.array(input_combinations)

C = input_matrix*C1 + (1-input_matrix)*C2

Rs = 50
RL = 50
Z_load = RL*np.ones(l, dtype=np.complex256) # initialisation


def update_Z_load(f, L0, C, Z_load):
    return Z_load/(f*Z_load*C+1) + 2*1j*f*L0

def find_ABCD(f, L0, C, Z_load, l):
    a = np.abs(-2*f**2*C*L0 + 1)
    b = np.abs(-f**2 *C*Z_load*L0 + 1j*f*L0)
    c = np.abs((1j*f*C)/(-2*f**2 *C*L0 + 1))
    d = np.abs(1j*C*Z_load + 1)

    Z0 = np.sqrt(L0/C)
    gamma = 1j*2*np.pi*f*np.sqrt(L0*C)
    a = Z0*np.sinh(gamma*1e-6)
    b = np.cosh(gamma*1e-6)
    c = b
    d = (1/Z0)*np.sinh(gamma*1e-6)
    return a, b, c, d

def find_S_matrix(a, b, c, d, f, L0, C, Z_load, l):
    Z0 = np.sqrt(L0/C)
    S11 = (a + b/Z0 - c/Z0 - d)/(a + b/Z0 + c/Z0 + d)
    S12 = 2 * (a*d - b*c) / (a + b/Z0 + c*Z0 + d)
    S21 = 2 / (a + b/Z0 + c*Z0 + d)
    S22 = (-a + b/Z0 - c*Z0 + d) / (a + b/Z0 + c*Z0 + d)
    return S11, S12, S21, S22

def plot_S_params(f, S11, S12, S21, S22):
    gridSize = int(np.ceil(np.sqrt(4)))
    fig_S_params, ax = plt.subplots(gridSize, gridSize, figsize=(10, 10))
    ax.flat[0].semilogx(f, 20*np.log10(np.abs(S11)))
    ax.flat[0].set_xlabel("f")
    ax.flat[0].set_ylabel("$S_{11}$")
    ax.flat[1].semilogx(f, 20*np.log10(np.abs(S12)))
    ax.flat[1].set_xlabel("f")
    ax.flat[1].set_ylabel("$S_{12}$")
    ax.flat[2].semilogx(f, 20*np.log10(np.abs(S21)))
    ax.flat[2].set_xlabel("f")
    ax.flat[2].set_ylabel("$S_{21}$")
    ax.flat[3].semilogx(f, 20*np.log10(np.abs(S22)))
    ax.flat[3].set_xlabel("f")
    ax.flat[3].set_ylabel("$S_{22}$")
    '''fig_S_params_lin, ax_lin = plt.subplots(gridSize, gridSize, figsize=(10, 10))
    ax_lin.flat[0].plot(f, 20*np.log10(np.abs(S11)))
    ax_lin.flat[0].set_xlabel("f")
    ax_lin.flat[0].set_ylabel("$S_{11}$")
    ax_lin.flat[1].plot(f, 20*np.log10(np.abs(S12)))
    ax_lin.flat[1].set_xlabel("f")
    ax_lin.flat[1].set_ylabel("$S_{12}$")
    ax_lin.flat[2].plot(f, 20*np.log10(np.abs(S21)))
    ax_lin.flat[2].set_xlabel("f")
    ax_lin.flat[2].set_ylabel("$S_{21}$")
    ax_lin.flat[3].plot(f, 20*np.log10(np.abs(S22)))
    ax_lin.flat[3].set_xlabel("f")
    ax_lin.flat[3].set_ylabel("$S_{22}$")'''
    plt.show()


for i in range(1):
    Z_load = RL*np.ones(l, dtype=np.complex256) # initialisation
    for j in range(num_of_sections):
        a, b, c, d = find_ABCD(f, L0, C[i][j], Z_load, l)
        S11_new, S12_new, S21_new, S22_new = find_S_matrix(a, b, c, d, f, L0, C[i][j], Z_load, l)
        if (j == 0):
            S11 = S11_new
            S21 = S21_new
            S12 = S12_new
            S22 = S22_new
        else:
            S11 = S11*S11_new + S12*S21_new
            S12 = S11*S12_new + S12*S22_new
            S21 = S21*S11_new + S22*S21_new
            S22 = S21*S12_new + S22*S22_new

        Z_load = update_Z_load(f, L0, C[i][j], Z_load)
    plot_S_params(f, S11, S12, S21, S22)
