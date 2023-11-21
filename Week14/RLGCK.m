% Define the coupled microstrip line
mline = coupledMicrostripLine;

% Generate S-parameters
frequencies = logspace(7, 10, 100);
spar = sparameters(mline, frequencies);

% Extract S-parameter data
sparam_data = spar.Parameters;

% Initialize a 3D array to store the S-parameter data
num_ports = size(sparam_data, 1);
num_freq = size(sparam_data, 3);
sparam_array = zeros(num_ports, num_ports, num_freq);

% Convert S-parameters to a 3D array
for k = 1:num_freq
    sparam_array(:, :, k) = sparam_data(:, :, k);
end

% Initialize matrices for SA and SB
SA = zeros(2, 2, num_freq);
SB = zeros(2, 2, num_freq);

% Extract SA and SB from sparam_array
for k = 1:num_freq
    SA(:, :, k) = sparam_array(1:2, 1:2, k);
    SB(:, :, k) = sparam_array(3:4, 3:4, k);
end

S_odd = SA + SB;
S_even = SA - SB;

% Convert S_odd and S_even to ABCD matrices
ABCD_odd = zeros(2, 2, num_freq);
ABCD_even = zeros(2, 2, num_freq);

for k = 1:num_freq
    ABCD_odd(:, :, k) = s2abcd(S_odd(:, :, k));
    ABCD_even(:, :, k) = s2abcd(S_even(:, :, k));
end

num_of_sections = 8; l = 4e-6;

[Z0_odd, gamma_odd] = ABCD2Z0Gamma(frequencies, l*num_of_sections, ABCD_odd);
[Z0_even, gamma_even] = ABCD2Z0Gamma(frequencies, l*num_of_sections, ABCD_even);


[R_odd, L_odd, G_odd, C_odd] = Z0Gamma2RLGC(Z0_odd, gamma_odd);
[R_even, L_even, G_even, C_even] = Z0Gamma2RLGC(Z0_even, gamma_even);


function [Z0, gamma] = ABCD2Z0Gamma(frequencies, l, ABCD)
    A11 = reshape(ABCD(1, 1, :), [1, length(frequencies)]);
    A22 = reshape(ABCD(2, 2, :), [1, length(frequencies)]);
    A12 = reshape(ABCD(1, 2, :), [1, length(frequencies)]);

    Z0 = sqrt(A11./A22);
    gamma = acosh(A12)./l;
end

function [R, L, G, C] = Z0Gamma2RLGC(Z0, gamma)

    R = real(gamma.*Z0);
    L = imag(gamma.*Z0);
    G = real(gamma./Z0);
    C = imag(gamma./Z0);
end

