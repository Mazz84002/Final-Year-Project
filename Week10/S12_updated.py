import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import constants
import random
import pandas as pd
import itertools

# ------------ Line parameters ---------------

L0 = 1e-7
C1 = 3e-9
C2 = 3e-10
# Generate all possible combinations of an 8-bit input
input_combinations = list(itertools.product([0, 1], repeat=8))
# Convert the combinations to A11 NumPy array
input_matrix = np.array(input_combinations)
C = input_matrix*C1 + (1-input_matrix)*C2

Rs = 50
RL = 50

# ---------- Simulation parameters ------------

fstart = 1e3
fstop = 100e9
fpoints = 100
f = np.logspace(np.log10(fstart), np.log10(fstop), fpoints)

num_of_sections = 8
num_of_combinations = 2**num_of_sections

# ----------- Helper functions -----------

def update_Z_load(f, L0, C, Z_load):
    return Z_load/(f*Z_load*C+1) + 2*1j*f*L0

def find_ABCD(f, L0, C, Z_load):
    A11 = np.abs(-2*f**2*C*L0 + 1)
    A12 = np.abs(-f**2 *C*Z_load*L0 + 1j*f*L0)
    A21 = np.abs((1j*f*C)/(-2*f**2 *C*L0 + 1))
    A22 = np.abs(1j*C*Z_load + 1)
    return A11, A12, A21, A22

def find_S_matrix(A11, A12, A21, A22, C, L0):
    Z0 = np.sqrt(L0/C)
    S11 = (A11 + A12/Z0 - A21/Z0 - A22)/(A11 + A12/Z0 + A21/Z0 + A22)
    S12 = 2 * (A11*A22 - A12*A21) / (A11 + A12/Z0 + A21*Z0 + A22)
    S21 = 2 / (A11 + A12/Z0 + A21*Z0 + A22)
    S22 = (-A11 + A12/Z0 - A21*Z0 + A22) / (A11 + A12/Z0 + A21*Z0 + A22)
    return S11, S12, S21, S22

def plot_S_params(f, S11, S12, S21, S22):
    gridSize = int(np.ceil(np.sqrt(4)))
    fig_S_params, ax = plt.subplots(gridSize, gridSize, figsize=(10, 10))
    ax.flat[0].plot(f, 20*np.log10(np.abs(S11)))
    ax.flat[1].plot(f, 20*np.log10(np.abs(S12)))
    ax.flat[2].plot(f, 20*np.log10(np.abs(S21)))
    ax.flat[3].plot(f, 20*np.log10(np.abs(S22)))
    plt.show()


# -------- Simulation of one section with constant Z0 -----------

i = 0
Z_load = RL*np.ones(fpoints, dtype=np.complex256) # initialisation
for j in range(num_of_sections):
    if (j == 0):
        A11, A12, A21, A22 = find_ABCD(f, L0, C2, Z_load)
    else:
        A11_new, A12_new, A21_new, A22_new = find_ABCD(f, L0, C2, Z_load)
        # matrix multiplication
        A11 = A11*A11_new + A12*A21_new
        A12 = A11*A12_new + A12*A22_new
        A21 = A21*A11_new + A22*A21_new
        A22 = A21*A12_new + A22*A22_new

    Z_load = update_Z_load(f, L0, C[i][j], Z_load)
    
S11, S12, S21, S22 = find_S_matrix(A11, A12, A21, A22, C2, L0)

plot_S_params(f, S11, S12, S21, S22)