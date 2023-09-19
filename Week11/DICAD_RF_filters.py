import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import itertools


def main():
    index = 0
    epsilon_0 = constants.epsilon_0

    fstart = -10
    fstop = 15
    fpoints = 1000
    f = np.logspace(fstart, fstop, fpoints)

    Lo, Co_no_shunt, Le, Ce_no_shunt, Ca, Cm, Cst = Z_eff_2_L_C(50, 50, 300*epsilon_0, 300*epsilon_0, 244.6e-6, 0.035e-6)

    # Assume negligible looses
    R = 1e-6
    Gm = 1e-8
    Ga = 1e-10

    l = 1e-4

    # ------------------ C matrix -------------------
    num_of_sections = 16
    num_of_combinations = 2**num_of_sections

    # Generate all possible combinations of an 8-bit input
    input_combinations = list(itertools.product([0, 1], repeat=num_of_sections))

    # Convert the combinations to a NumPy array
    input_matrix = np.array(input_combinations)

    C_odd = input_matrix*(Co_no_shunt+Cst) + (1-input_matrix)*Co_no_shunt # capacitance values for every single state
    C_even = input_matrix*(Ce_no_shunt) + (1-input_matrix)*Ce_no_shunt # capacitance values for every single state


    # ----------- S calulator ---------------
    S = np.zeros((4, 4, len(f)), dtype=complex)

    for j in range(num_of_sections):
        Lo, Co, Go, Ro, gamma_o, Z0o, theta_o = odd_mode_params(f, Lo, C_odd[index][num_of_sections - j - 1], Ca, Gm, Ga, R)
        Le, Ce, Ge, Re, gamma_e, Z0e, theta_e = even_mode_params(f, Le, C_even[index][num_of_sections - j - 1], Ca, Gm, Ga, R)

        cot_theta_o = 1/np.tan(theta_o)
        cot_theta_e = 1/np.tan(theta_e)

        csc_theta_o = 1/np.sin(theta_o)
        csc_theta_e = 1/np.sin(theta_e)

        Z = create_Z_matrix(Z0e, cot_theta_e, csc_theta_e, Z0o, cot_theta_o, csc_theta_o)

        F = create_F(50*np.ones(len(f)), 50*np.ones(len(f)), 50*np.ones(len(f)), 50*np.ones(len(f)))
        G = create_G(50*np.ones(len(f)), 50*np.ones(len(f)), 50*np.ones(len(f)), 50*np.ones(len(f)))
        S_curr = z2s(Z, F, G) # convert to S parameters for cascading

        if (j == 0):
            S = S_curr
        else:
            S = multiply_mat(S_curr, S)
        
    # update F and G
    F = create_F(50*np.ones(len(f)), 50*np.ones(len(f)), 50*np.ones(len(f)), 50*np.ones(len(f)))
    G = create_G(50*np.ones(len(f)), 50*np.ones(len(f)), 50*np.ones(len(f)), 50*np.ones(len(f)))

    Z = s2z(S, F, G)

    # V2 = 0 and I3 = 0 - reduces the matrix to that of a 2 port system

    Z_lpf = reduce_low_pass(Z)
    S_lpf = z2s(Z_lpf, create_reduced_F(f), create_reduced_G(f))

    plot_S_params(f, S_lpf[0, 0, :], S_lpf[0, 1, :], S_lpf[1, 0, :], S_lpf[1, 1, :])



## --------------- Line parameters -----------------

def Z_eff_2_L_C(Z0o, Z0e, eps_o, eps_e, L, T):
    c = constants.speed_of_light

    Lo = (Z0o/c)*np.sqrt(eps_o)
    Co_no_shunt = np.sqrt(eps_o)/(c*Z0o)

    Le = (Z0e/c)*np.sqrt(eps_e)
    Ce_no_shunt = np.sqrt(eps_e)/(c*Z0e)
    
    Ca = Ce_no_shunt
    Cm = (Co_no_shunt+Ce_no_shunt)/2

    Cst = (constants.epsilon_0*T)/(10e-6) # shunt capacitance

    return Lo, Co_no_shunt, Le, Ce_no_shunt, Ca, Cm, Cst


## -------------------- Plotters --------------------

def plot_S_params(f, S11, S12, S21, S22):
    gridSize = int(np.ceil(np.sqrt(4)))
    fig_S_params, ax = plt.subplots(gridSize, gridSize, figsize=(10, 10))
    ax.flat[0].semilogx(f, 20*np.log10(np.abs(S11)))
    ax.flat[0].set_xlabel("f")
    ax.flat[0].set_ylabel("$S_{11}$")
    ax.flat[0].grid(True)  # Add grid lines

    ax.flat[1].semilogx(f, 20*np.log10(np.abs(S12)))
    ax.flat[1].set_xlabel("f")
    ax.flat[1].set_ylabel("$S_{12}$")
    ax.flat[1].grid(True)  # Add grid lines

    ax.flat[2].semilogx(f, 20*np.log10(np.abs(S21)))
    ax.flat[2].set_xlabel("f")
    ax.flat[2].set_ylabel("$S_{21}$")
    ax.flat[2].grid(True)  # Add grid lines

    ax.flat[3].semilogx(f, 20*np.log10(np.abs(S22)))
    ax.flat[3].set_xlabel("f")
    ax.flat[3].set_ylabel("$S_{22}$")
    ax.flat[3].grid(True)  # Add grid lines
    plt.show()
# ------------------ Matrix Calculators -------------

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

# --------------- odd and even parameters -------------

def odd_mode_params(f, Lo, Cm, Ca, Gm, Ga, R):
    w = 2*np.pi*f
    Co = 2*Cm - Ca
    Go = 2*Gm + Ga
    Lo = Lo
    Ro = R
    #gamma_o = 1e-5 + 1j*(2*np.pi*w*np.sqrt(Lo*Co))
    gamma_o = np.sqrt( (1j*w*Lo + Ro)*(1j*w*(Co) + Go) )
    Z0o = np.sqrt((Ro+1j*w*Lo)/(Go + 1j*w*Co))


    f0_o = 1/(2*np.pi*np.sqrt(Lo/Co)) # frequency at which the coupled lines are a quarter-wavelength long electrically when excited in the odd mode
    theta_o = (np.pi/2)*(f/f0_o)

    return Lo, Co, Go, Ro, gamma_o, Z0o, theta_o

def even_mode_params(f, Le, Cm, Ca, Gm, Ga, R):
    w = 2*np.pi*f
    Ce = Ca
    Ge = Ga
    Le = Le
    Re = R
    #gamma_e = 1e-4 + 1j*(2*np.pi*w*np.sqrt(Le*Ce))
    gamma_e = np.sqrt( (1j*w*Le + Re)*(1j*w*(Ce) + Ge) )
    Z0e = np.sqrt((Re+1j*w*Le)/(Ge + 1j*w*Ce))

    f0_e = 1/(2*np.pi*np.sqrt(Le/Ce)) #frequency at which the coupled lines are a quarter-wavelength long electrically when excited in the even mode,
    theta_e = (np.pi/2)*(f/f0_e)

    return Le, Ce, Ge, Re, gamma_e, Z0e, theta_e


# --------------------- Z and S parameters matrices -----------------

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

def reduce_low_pass(Z):
    return Z[np.ix_([0, 3], [0, 3])]

if __name__ == "__main__":
  main()