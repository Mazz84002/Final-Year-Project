import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy import constants

# Set a global line width for all plots
plt.rcParams['lines.linewidth'] = 0.5

def main():

    # ---------------------- RLGCK Parameters -------------------------

    fstart = 1e9
    fstop = 0.5e11
    fpoints = 500
    f = np.linspace(fstart, fstop, fpoints)
    l = 6e-6 / 16

    R_odd = np.linspace(0.8e6, 3.2e6, fpoints)
    L_odd = (0.3 * np.exp(-f / f[0]) + 3.8) * 1e-7
    G_odd = np.linspace(0, 60e-1, fpoints)
    C_odd = np.linspace(1.5e-10, 1.4e-10, fpoints)

    R_even = R_odd/1.3
    L_even = L_odd/1.1
    G_even = G_odd/1.5
    C_even = C_odd/1.5
    Cst = 10*C_odd

    # ------------------ Create Z and ABCD Matrix -----------------------
    gamma_odd_off, gamma_even_off, Z0_odd_off, Z0_even_off, theta_odd_off, theta_even_off, \
    gamma_odd_on, gamma_even_on, Z0_odd_on, Z0_even_on, theta_odd_on, theta_even_on = gamma_Z0(R_odd, L_odd, G_odd, C_odd, R_even, L_even, G_even, C_even, Cst, f, l)

    Z_on, Z_off = Zparameters10(Z0_odd_off, Z0_even_off, Z0_odd_on, Z0_even_on, theta_odd_off, theta_even_off, theta_odd_on, theta_even_on)
    """plot_Z_params(f, Z_off, "Z_off")
    plot_Z_params(f, Z_on, "Z_on")"""
    ABCD_off, ABCD_on = Z2ABCDmultiport(Z_off, Z_on, fpoints)
    plot_ABCD(ABCD_off, f, "ABCD off")
    plot_ABCD(ABCD_on, f, "ABCD on")

    # ---------------- Creating Filters ------------------
    num_of_sections = 8
    num_of_combinations = 2**num_of_sections
    input_combinations = list(itertools.product([0, 1], repeat=num_of_sections))
    input_matrix = np.array(input_combinations)

    # Initialize a list to store the identity matrices
    identity_matrices = []
    # Create identity matrices and store them in the list
    for i in range(fpoints):
        identity_matrices.append(np.eye(4))
    # Stack the identity matrices along the third dimension
    I = np.stack(identity_matrices, axis=-1)
    ABCD = I
    S_storage = {}
    S1_storage = {}
    S2_storage = {}
    S3_storage = {}

    gamma_odd_storage = {}
    gamma_even_storage = {}

    n = 16

    for index in range(num_of_combinations):
        for j in range(num_of_sections):
            if (input_matrix[index, j] == 1):
                ABCD = multiply_mat(ABCD, ABCD_on)
            else:
                ABCD = multiply_mat(ABCD, ABCD_off)
        Z, S = ABCD2ZS(ABCD, fpoints)
        S_storage[index] = S
        Z0_odd, gamma_odd, Z0_even, gamma_even = S2RLGCK4port(S, f, fpoints, l*num_of_sections, fig=1 if index == 5 else 0)
        gamma_odd_storage[index] = gamma_odd
        gamma_even_storage[index] = gamma_even

        S1, S2, S3 = port_reductions(Z, fpoints)
        S1_storage[index] = S1
        S2_storage[index] = S2
        S3_storage[index] = S3

    
    plot_S_parameters(f, S_storage, "S-parameters", n, smooth_window=1)

    # ---------------- 2-port reductions ---------------------
    plot_reduced_S_parameters(f, S1_storage, "Configuration 1", n, smooth_window=1)
    plot_reduced_S_parameters(f, S2_storage, "Configuration 2", n, smooth_window=1)
    plot_reduced_S_parameters(f, S3_storage, "Configuration 3", n, smooth_window=1)

    find_eps_eff(gamma_odd_storage, gamma_even_storage, f, 16)

    plt.show()

def gamma_Z0(R_odd, L_odd, G_odd, C_odd, R_even, L_even, G_even, C_even, Cst, f, l):
    w = 2 * np.pi * f

    gamma_odd_off = np.sqrt((1j * w * L_odd + R_odd) * (1j * w * C_odd + G_odd))
    gamma_even_off = np.sqrt((1j * w * L_even + R_even) * (1j * w * C_even + G_even))
    Z0_odd_off = np.sqrt((R_odd + 1j * w * L_odd) / (G_odd + 1j * w * C_odd))
    Z0_even_off = np.sqrt((R_even + 1j * w * L_even) / (G_even + 1j * w * C_even))
    theta_odd_off = gamma_odd_off * l
    theta_even_off = gamma_even_off * l

    gamma_odd_on = np.sqrt((1j * w * L_odd + R_odd) * (1j * w * (C_odd + Cst) + G_odd))
    gamma_even_on = np.sqrt((1j * w * L_even + R_even) * (1j * w * C_even + G_even))
    Z0_odd_on = np.sqrt((R_odd + 1j * w * L_odd) / (G_odd + 1j * w * (C_odd + Cst)))
    Z0_even_on = np.sqrt((R_even + 1j * w * L_even) / (G_even + 1j * w * C_even))
    theta_odd_on = gamma_odd_on * l
    theta_even_on = gamma_even_on * l

    return (gamma_odd_off, gamma_even_off, Z0_odd_off, Z0_even_off, theta_odd_off, theta_even_off,
            gamma_odd_on, gamma_even_on, Z0_odd_on, Z0_even_on, theta_odd_on, theta_even_on)

def create_Z_matrix(Z0_even, csc_theta_even, cot_theta_even, Z0_odd, csc_theta_odd, cot_theta_odd):
    Z11 = -(1j/2) * (Z0_even * cot_theta_even + Z0_odd * cot_theta_odd)
    Z12 = -(1j/2) * (Z0_even * cot_theta_even - Z0_odd * cot_theta_odd)
    Z13 = -(1j/2) * (Z0_even * csc_theta_even - Z0_odd * csc_theta_odd)
    Z14 = -(1j/2) * (Z0_even * csc_theta_even + Z0_odd * csc_theta_odd)

    Z = np.zeros((4, 4, len(Z11)), dtype=complex)

    Z[0, 0, :] = Z11
    Z[1, 1, :] = Z11
    Z[2, 2, :] = Z11
    Z[3, 3, :] = Z11

    Z[0, 1, :] = Z12
    Z[1, 0, :] = Z12
    Z[2, 3, :] = Z12
    Z[3, 2, :] = Z12

    Z[0, 2, :] = Z13
    Z[2, 0, :] = Z13
    Z[1, 3, :] = Z13
    Z[3, 1, :] = Z13

    Z[0, 3, :] = Z14
    Z[3, 0, :] = Z14
    Z[1, 2, :] = Z14
    Z[2, 1, :] = Z14

    return Z

def Zparameters10(Z0_odd_off, Z0_even_off, Z0_odd_on, Z0_even_on, theta_odd_off, theta_even_off, theta_odd_on, theta_even_on):
    cot_theta_odd_off = 1 / np.tan(theta_odd_off)
    cot_theta_odd_on = 1 / np.tan(theta_odd_on)
    cot_theta_even_off = 1 / np.tan(theta_even_off)
    cot_theta_even_on = 1 / np.tan(theta_even_on)

    csc_theta_odd_off = 1 / np.sin(theta_odd_off)
    csc_theta_odd_on = 1 / np.sin(theta_odd_on)
    csc_theta_even_off = 1 / np.sin(theta_even_off)
    csc_theta_even_on = 1 / np.sin(theta_even_on)

    Z_off = create_Z_matrix(Z0_even_off, csc_theta_even_off, cot_theta_even_off, Z0_odd_off, csc_theta_odd_off, cot_theta_odd_off)
    Z_on = create_Z_matrix(Z0_even_on, csc_theta_even_on, cot_theta_even_on, Z0_odd_on, csc_theta_odd_on, cot_theta_odd_on)

    return Z_on, Z_off

def plot_Z_params(f, Z, title):
    gridSize = int(np.ceil(np.sqrt(16)))
    fig_Z_params, axs = plt.subplots(gridSize, gridSize, figsize=(12, 9))

    for i in range(16):
        ax = axs[i // gridSize, i % gridSize]
        ax.semilogx(f, np.real(Z[i // 4, i % 4, :]))
        ax.grid(True)

    fig_Z_params.suptitle(title, fontsize=16)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

def invert_mat(A):
    M, N, P = A.shape
    inv_A = np.zeros((M, N, P), dtype=complex)
    for i in range(P):
        inv_A[:, :, i] = np.linalg.inv(A[:, :, i])
    return inv_A

def multiply_mat(A, B):
    M, N, P = A.shape
    C = np.zeros((M, N, P), dtype=complex)
    for i in range(P):
        C[:, :, i] = np.dot(A[:, :, i], B[:, :, i])
    return C

def Z2ABCDmultiport(Z_off, Z_on, fpoints):
    Zee_off = Z_off[0:2, 0:2, :]
    Zei_off = Z_off[0:2, 2:4, :]
    Zie_off = Z_off[2:4, 0:2, :]
    Zii_off = Z_off[2:4, 2:4, :]

    Zee_on = Z_on[0:2, 0:2, :]
    Zei_on = Z_on[0:2, 2:4, :]
    Zie_on = Z_on[2:4, 0:2, :]
    Zii_on = Z_on[2:4, 2:4, :]

    ABCD_off = np.zeros((4, 4, fpoints), dtype=complex)
    ABCD_on = np.zeros((4, 4, fpoints), dtype=complex)

    ABCD_off[0:2, 0:2, :] = multiply_mat(Zee_off, invert_mat(Zie_off))
    ABCD_off[0:2, 2:4, :] = multiply_mat(multiply_mat(Zee_off, invert_mat(Zie_off)), Zii_off) - Zei_off
    ABCD_off[2:4, 0:2, :] = invert_mat(Zie_off)
    ABCD_off[2:4, 2:4, :] = multiply_mat(invert_mat(Zie_off), Zii_off)

    ABCD_on[0:2, 0:2, :] = multiply_mat(Zee_on, invert_mat(Zie_on))
    ABCD_on[0:2, 2:4, :] = multiply_mat(multiply_mat(Zee_on, invert_mat(Zie_on)), Zii_on) - Zei_on
    ABCD_on[2:4, 0:2, :] = invert_mat(Zie_on)
    ABCD_on[2:4, 2:4, :] = multiply_mat(invert_mat(Zie_on), Zii_on)

    return ABCD_off, ABCD_on

def create_G0(Z1, Z2, Z3, Z4):
    G0 = np.zeros((4, 4, len(Z1)), dtype=Z1.dtype)
    G0[0, 0, :] = 1.0 / np.abs(np.sqrt(np.real(Z1)))
    G0[1, 1, :] = 1.0 / np.abs(np.sqrt(np.real(Z2)))
    G0[2, 2, :] = 1.0 / np.abs(np.sqrt(np.real(Z3)))
    G0[3, 3, :] = 1.0 / np.abs(np.sqrt(np.real(Z4)))
    return G0

def create_Z0(Z1, Z2, Z3, Z4):
    Z0 = np.zeros((4, 4, len(Z1)), dtype=Z1.dtype)
    Z0[0, 0, :] = Z1
    Z0[1, 1, :] = Z2
    Z0[2, 2, :] = Z3
    Z0[3, 3, :] = Z4
    return Z0

def ABCD2ZS(ABCD, fpoints):
    Z = np.zeros((4, 4, fpoints), dtype=ABCD.dtype)
    S = Z.copy()
    Zin = 50 * np.ones(fpoints)
    G0 = create_G0(Zin, Zin, Zin, Zin)
    Z0 = create_Z0(Zin, Zin, Zin, Zin)

    A = ABCD[0:2, 0:2, :]
    B = ABCD[0:2, 2:4, :]
    C = ABCD[2:4, 0:2, :]
    D = ABCD[2:4, 2:4, :]

    Z[0:2, 0:2, :] = multiply_mat(A, invert_mat(C))
    Z[0:2, 2:4, :] = multiply_mat(multiply_mat(A, invert_mat(C)), D) - B
    Z[2:4, 0:2, :] = invert_mat(C)
    Z[2:4, 2:4, :] = multiply_mat(invert_mat(C), D)

    S = multiply_mat(G0, multiply_mat(Z - np.conj(Z0), multiply_mat(invert_mat(Z + Z0), invert_mat(G0))))
    
    return Z, S

def moving_average(data, window_size):
    """Calculate the moving average of a 1D array."""
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def plot_S_parameters(f, S_storage, title, n, smooth_window=1):
    N = len(S_storage)
    gridSize = int(np.ceil(np.sqrt(16)))
    fig_S_params, ax = plt.subplots(gridSize, gridSize, figsize=(15, 9))

    # Create a list of legend entries
    legend_entries = []
    for j in range(0, N, int(N/n)):
        legend_entries.append(f"index={format(j, f'0{8}b')}")

    # Add a legend to each subplot
    for i in range(16):
        for j in range(0, N, int(N/n)):
            # Smooth the S-parameter data
            smoothed_S = moving_average(20*np.log10(np.abs(S_storage[j][int(i/4), i%4, :])), smooth_window)
            line, = ax.flat[i].semilogx(f[:len(smoothed_S)], smoothed_S)
        ax.flat[i].grid(True)  # Add grid lines

    # Set a common title for all subplots
    fig_S_params.suptitle(title, fontsize=16)
    fig_S_params.legend(legend_entries)

def port_reductions(Z, fpoints):
    # 1
    Z1 = np.delete(Z, [1, 2], axis=0)
    Z1 = np.delete(Z1, [1, 2], axis=1)

    G0 = np.zeros((2, 2, fpoints), dtype=Z.dtype)
    G0[0, 0, :] = 1.0 / np.abs(np.sqrt(50*np.ones(fpoints)))
    G0[1, 1, :] = 1.0 / np.abs(np.sqrt(50*np.ones(fpoints)))
    Z0 = np.zeros((2, 2, fpoints), dtype=Z.dtype)
    Z0[0, 0, :] = 50*np.ones(fpoints)
    Z0[1, 1, :] = 50*np.ones(fpoints)

    S1 = multiply_mat(G0, multiply_mat(Z1 - np.conj(Z0), multiply_mat(invert_mat(Z1 + Z0), invert_mat(G0))))

    # 2
    Z2 = np.delete(Z, [1, 3], axis=0)
    Z2 = np.delete(Z2, [1, 3], axis=1)
    S2= multiply_mat(G0, multiply_mat(Z2 - np.conj(Z0), multiply_mat(invert_mat(Z2 + Z0), invert_mat(G0))))

    # 3
    Z3 = np.delete(Z, [2, 3], axis=0)
    Z3 = np.delete(Z3, [2, 3], axis=1)
    S3= multiply_mat(G0, multiply_mat(Z3 - np.conj(Z0), multiply_mat(invert_mat(Z3 + Z0), invert_mat(G0))))

    return S1, S2, S3

def plot_reduced_S_parameters(f, S_storage, title, n, smooth_window=1):
    N = len(S_storage)
    gridSize = int(np.ceil(np.sqrt(4)))
    fig_S_params, ax = plt.subplots(gridSize, gridSize, figsize=(12, 5))

    # Create a list of legend entries
    legend_entries = []
    for j in range(0, N, int(N/n)):
        legend_entries.append(f"index={format(j, f'0{8}b')}")

    # Add a legend to each subplot
    for i in range(4):
        for j in range(0, N, int(N/n)):
            # Smooth the S-parameter data
            smoothed_S = moving_average(20*np.log10(np.abs(S_storage[j][int(i/2), i%2, :])), smooth_window)
            line, = ax.flat[i].semilogx(f[:len(smoothed_S)], smoothed_S)
        ax.flat[i].grid(True)  # Add grid lines

    # Set a common title for all subplots
    fig_S_params.suptitle(title, fontsize=16)
    fig_S_params.legend(legend_entries)

def S2ABCD(S, fpoints):
    identity_matrices = []
    if len(S[:, 0, 0]) == 4:
        Zin = 50 * np.ones(fpoints)
        G0 = create_G0(Zin, Zin, Zin, Zin)
        Z0 = create_Z0(Zin, Zin, Zin, Zin)
        for i in range(fpoints):
            identity_matrices.append(np.eye(4))
        I = np.stack(identity_matrices, axis=-1)
    else:
        G0 = np.zeros((2, 2, fpoints), dtype=S.dtype)
        G0[0, 0, :] = 1.0 / np.abs(np.sqrt(50*np.ones(fpoints)))
        G0[1, 1, :] = 1.0 / np.abs(np.sqrt(50*np.ones(fpoints)))
        Z0 = np.zeros((2, 2, fpoints), dtype=S.dtype)
        Z0[0, 0, :] = 50*np.ones(fpoints)
        Z0[1, 1, :] = 50*np.ones(fpoints)
        for i in range(fpoints):
            identity_matrices.append(np.eye(2))
        I = np.stack(identity_matrices, axis=-1)
    
    # S -> Z
    Z = multiply_mat(multiply_mat(multiply_mat(invert_mat(G0), invert_mat(I-S)), multiply_mat(S, Z0) + np.conj(Z0)), G0)

    if len(S[:, 0, 0]) == 4:
        Zee = Z[0:2, 0:2, :]
        Zei = Z[0:2, 2:4, :]
        Zie = Z[2:4, 0:2, :]
        Zii = Z[2:4, 2:4, :]

        ABCD = np.zeros((4, 4, fpoints), dtype=S.dtype)
        ABCD[0:2, 0:2, :] = multiply_mat(Zee, invert_mat(Zie))
        ABCD[0:2, 2:4, :] = multiply_mat(multiply_mat(Zee, invert_mat(Zie)), Zii) - Zei
        ABCD[2:4, 0:2, :] = invert_mat(Zie)
        ABCD[2:4, 2:4, :] = multiply_mat(invert_mat(Zie), Zii)
    else:
        Zee = Z[0, 0, :]
        Zei = Z[0, 1, :]
        Zie = Z[1, 0, :]
        Zii = Z[1, 1, :]

        ABCD = np.zeros((2, 2, fpoints), dtype=S.dtype)

        ABCD[0, 0, :] = Zee/Zie
        ABCD[0, 1, :] = Zee*Zii/Zie - Zei
        ABCD[1, 0, :] = 1/Zie
        ABCD[1, 1, :] = Zii/Zie

    return Z, ABCD

def plot_ABCD(ABCD, f, title):
    # Extracting matrices A, B, C, D
    A = ABCD[0, 0, :]
    B = ABCD[0, 1, :]
    C = ABCD[1, 0, :]
    D = ABCD[1, 1, :]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    axes[0].semilogx(f, np.real(A))
    axes[0].semilogx(f, np.imag(A))
    axes[0].grid(True)
    axes[1].semilogx(f, np.real(B))
    axes[1].semilogx(f, np.imag(B))
    axes[1].grid(True)
    axes[2].semilogx(f, np.real(C))
    axes[2].semilogx(f, np.imag(C))
    axes[2].grid(True)
    axes[3].semilogx(f, np.real(D))
    axes[3].semilogx(f, np.imag(D))
    axes[3].grid(True)

    fig.suptitle(title, fontsize=16)

def S2RLGCK4port(S, f, fpoints, L, fig=0):
    SA = S[0:2, 0:2, :]
    SB = S[0:2, 2:4, :]
    S_even = SA - SB
    S_odd = SA + SB

    garbage, ABCD_odd = S2ABCD(S_odd, fpoints)
    garbage, ABCD_even = S2ABCD(S_even, fpoints)

    Z0_odd = np.sqrt(ABCD_odd[0, 0, :]/ ABCD_odd[1, 1, :])
    gamma_odd = np.arccosh(ABCD_odd[0, 1, :])/L
    Z0_even = np.sqrt(ABCD_even[0, 0, :]/ ABCD_even[1, 1, :])
    gamma_even = np.arccosh(ABCD_even[0, 1, :])/L

    if fig == 1:
        plot_ABCD(ABCD_odd, f, "ABCD odd")
        plot_ABCD(ABCD_even, f, "ABCD even")

        gridSize = int(np.ceil(np.sqrt(4)))
        fig_RLGCK, ax = plt.subplots(gridSize, gridSize, figsize=(12, 5))
        ax.flat[0].loglog(f, np.real(Z0_odd))
        ax.flat[0].loglog(f, np.imag(Z0_odd))
        ax.flat[0].grid(True)  # Add grid lines

        ax.flat[1].loglog(f, np.real(gamma_odd))
        ax.flat[1].loglog(f, np.imag(gamma_odd))
        ax.flat[1].grid(True)  # Add grid lines

        ax.flat[2].loglog(f, np.real(Z0_even))
        ax.flat[2].loglog(f, np.imag(Z0_even))
        ax.flat[2].grid(True)  # Add grid lines

        ax.flat[3].loglog(f, np.real(gamma_even))
        ax.flat[3].loglog(f, np.imag(gamma_even))
        ax.flat[3].grid(True)  # Add grid lines

        fig_RLGCK.suptitle("Z0 ang Gamma for odd mode", fontsize=16)

    return Z0_odd, gamma_odd, Z0_even, gamma_even

def find_eps_eff(gamma_odd_storage, gamma_even_storage, f, n): # takes in arrays for all combinations
    #eps_odd = np.square((np.imag(gamma_odd)*3e8)/(2*np.pi*f))
    #eps_even = np.square((np.imag(gamma_even)*3e8)/(2*np.pi*f))
    N = len(gamma_odd_storage)
    fig_eps, ax = plt.subplots(2, 1, figsize=(6, 9))
    for j in range(0, N, int(N/n)):
        eps_odd = np.square((np.imag(gamma_odd_storage[j])*3e8)/(2*np.pi*f))
        eps_even = np.square((np.imag(gamma_even_storage[j])*3e8)/(2*np.pi*f))
        line, = ax.flat[0].semilogx(f, eps_odd)
        ax.flat[0].grid(True)  # Add grid lines
        line, = ax.flat[1].semilogx(f, eps_even)
        ax.flat[1].grid(True)  # Add grid lines
    
    legend_entries = []
    for j in range(0, N, int(N/n)):
        legend_entries.append(f"index={format(j, f'0{8}b')}")

    fig_eps.suptitle("eps", fontsize=16)
    fig_eps.legend(legend_entries)

if __name__ == main():
    main()