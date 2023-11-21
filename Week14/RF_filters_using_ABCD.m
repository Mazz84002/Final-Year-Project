clear all; close all;

%% constants

epsilon_0 = 8.85e-12;

fstart = 7;
fstop = 10;
fpoints = 100;
f = logspace(fstart, fstop, fpoints);

tstart = 0;
tstop = 5e-6;
tpoints = 500;
t = linspace(tstart, tstop, tpoints);
t = reshape(t, [length(t), 1]);

%% main

num_of_sections = 8;
num_of_combinations = 2^num_of_sections;

l = 4e-6;

input_matrix = create_input_matrix(num_of_sections);
%[L_odd, C_odd, L_even, C_even, Ca, Cm, Cst, R, Gm, Ga] = initialise_parameters(f, fpoints);


%[L_odd, C_odd, G_odd, R_odd, gamma_o, Z0_odd, theta_odd] = odd_mode_params(f, L_odd, Cm, Ca, Gm, Ga, R);
%[L_even, C_even, G_even, R_even, gamma_e, Z0_even, theta_even] = even_mode_params(f, L_even, Cm, Ca, Gm, Ga, R);



% Define the coupled microstrip line
mline = coupledMicrostripLine;

% Generate S-parameters
frequencies = f;
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


[Z0o_off, Z0o_on, Z0e_off, Z0e_on, gamma_o_off, gamma_o_on, gamma_e_off, gamma_e_on, ... 
    theta_o_off, theta_o_on, theta_e_off, theta_e_on] ... 
    = Z0_even_odd_on_off(f, L_odd, L_even, (C_odd+C_even)/2, C_even, C_odd*10, (G_odd-G_even)/2, G_even, R_odd, R_even);



% Plot both `off` and `on` ABCD parameters
[Ao_off, Ae_off, Ao_on, Ae_on] = create_on_off_A_matrices(f, L_odd, L_even, (C_odd+C_even)/2, C_even, C_odd*10, (G_odd-G_even)/2, G_even, R_odd, R_even, l);
plot_full_ABCD_params(f, Ao_off, Ae_off, 'ABCD - off');
plot_full_ABCD_params(f, Ao_on, Ae_on, 'ABCD - on');


% Plot all Zins
[Zin_o_off, Zin_o_on, Zin_e_off, Zin_e_on] ... 
    = find_all_Zins(Z0o_off, Z0o_on, Z0e_off, Z0e_on, gamma_o_off, gamma_o_on, gamma_e_off, gamma_e_on, l, 50);
plot_possible_Zins(f, Zin_o_off, Zin_o_on, Zin_e_off, Zin_e_on);


% Plot both `off` and `on` S parameters
Zin = 50*ones(1, length(f));
S_off = a2sw(Ao_off, Ae_off, Zin, Zin, Zin, Zin);
S_on = a2sw(Ao_on, Ae_on, Zin, Zin, Zin, Zin);
plot_full_S_params(f, S_off, 'S - off');
plot_full_S_params(f, S_on, 'S - on');


% Initialize a cell array to store the identity matrices
identity_matrices = cell(1, fpoints);
% Create identity matrices and store them in the cell array
for i = 1:fpoints
    identity_matrices{i} = eye(4);
end

% Create a cell array to store S matrices
S_cell_array = cell(num_of_combinations, 1);

Cst = C_odd*10;

for index = 1:1:num_of_combinations
    S = cat(3, identity_matrices{:}); % initialse S matrix
    for j = 1:num_of_sections % iterate aross sections
        if (input_matrix(index, j) == 1)
            [Z01_odd, Z02_odd, Z03_odd, Z04_odd] = find_Zin_for_curr_section(num_of_sections, j, find_Zc(f, R_odd, L_odd, G_odd, C_odd+Cst), gamma_o_on, l, f);
            [Z01_even, Z02_even, Z03_even, Z04_even] = find_Zin_for_curr_section(num_of_sections, j, find_Zc(f, R_even, L_even, G_even, C_even), gamma_e_on, l, f);

            S_curr = a2sw(Ao_off, Ae_off, Z01_odd, Z03_odd, Z01_even, Z03_even);
            S = multiply_mat(S, S_curr);
    
        else
            [Z01_odd, Z02_odd, Z03_odd, Z04_odd] = find_Zin_for_curr_section(num_of_sections, j, find_Zc(f, R_odd, L_odd, G_odd, C_odd), gamma_o_on, l, f);
            [Z01_even, Z02_even, Z03_even, Z04_even] = find_Zin_for_curr_section(num_of_sections, j, find_Zc(f, R_even, L_even, G_even, C_even), gamma_e_on, l, f);
    
            S_curr = a2sw(Ao_off, Ae_off, Z01_odd, Z03_odd, Z01_even, Z03_even);
            S = multiply_mat(S, S_curr);
        end
    end
    % Store the calculated S matrix in the cell array
    S_cell_array{index} = S;
end

plot_full_S_params(f, S_cell_array{128}, 'S - 128');
[eps_odd, eps_even] = eps_w_from_S_parameters(f, l, num_of_sections,  S_cell_array{128});
plot_eps_ft(f, t, S_cell_array{128}, eps_odd, eps_even);


%% matrix operations

function inv_A = invert_mat(A)
    [M, N, P] = size(A);
    inv_A = complex(zeros(M, N, P), zeros(M, N, P));
    
    for i = 1:P
        inv_A(:, :, i) = inv(A(:, :, i));
    end
end

function C = multiply_mat(A, B)
    [M, N, P] = size(A);
    C = complex(zeros(M, N, P), zeros(M, N, P));
    
    for i = 1:P
        C(:, :, i) = A(:, :, i) * B(:, :, i);
    end
end


%% parameters setup

function input_matrix = create_input_matrix(num_of_sections)

    num_of_combinations = 2^num_of_sections;
    
    % Generate all possible combinations of a 16-bit input
    input_combinations = dec2bin(0:(num_of_combinations-1), num_of_sections) - '0';
    
    % Convert the combinations to a MATLAB matrix (array)
    input_matrix = input_combinations;

end

function [L_odd, C_odd, L_even, C_even, Ca, Cm, Cst, R, Gm, Ga] = initialise_parameters(f, fpoints)
   % Define the parameters of the quadratic function
    a = 4.0;  % Coefficient for the quadratic term
    b = 1.0;  % Coefficient for the linear term
    c = 4.0e-7;  % Coefficient for the constant term
    x = linspace(6e-7, 4e-7, fpoints);  % You can adjust the range as needed
    L = a * x.^2 + b * x + c;
    
    k = 0.66;
    L_odd = L * (1 + k);
    L_even = L * (1 - k);
    
    C_odd = 0.6e-10;
    C_even = C_odd/1.2;
    
    Ca = C_even;
    Cm = C_odd;
    
    Cst = 10 * C_even;
    
    a = 4.0;  % Coefficient for the quadratic term
    b = 1.0;  % Coefficient for the linear term
    c = 4.0;  % Coefficient for the constant term
    x = linspace(1e4, 1e6, fpoints);  % You can adjust the range as needed
    R = a * x.^2 + b * x + c;
    
    Gm = linspace(10, 1e5, fpoints);
    Ga = linspace(10, 1e5, fpoints);
end


function [L_odd, C_odd, G_odd, R_odd, gamma_o, Z0_odd, theta_odd] = odd_mode_params(f, L_odd, Cm, Ca, Gm, Ga, R)
    w = 2 * pi * f;
    C_odd = 2 * Cm - Ca;
    G_odd = 2 * Gm + Ga;
    R_odd = R;
    
    gamma_o = sqrt((1i .* w .* L_odd + R_odd) .* (1i .* w .* C_odd + G_odd));
    Z0_odd = sqrt((R_odd + 1i .* w .* L_odd) ./ (G_odd + 1i .* w .* C_odd));
    
    f0_o = 1 ./ (2 .* pi .* sqrt(L_odd .* C_odd));
    theta_odd = (pi ./ 2) .* (f ./ f0_o);
end


function [L_even, C_even, G_even, R_even, gamma_e, Z0_even, theta_even] = even_mode_params(f, L_even, Cm, Ca, Gm, Ga, R)
    w = 2 * pi * f;
    C_even = Ca;
    G_even = Ga;
    R_even = R;
    
    gamma_e = sqrt((1i .* w .* L_even + R_even) .* (1i .* w .* C_even + G_even));
    Z0_even = sqrt((R_even + 1i .* w .* L_even) ./ (G_even + 1i .* w .* C_even));
    
    f0_e = 1 ./ (2 .* pi .* sqrt(L_even .* C_even));
    theta_even = (pi ./ 2) .* (f ./ f0_e);
end


%% Input Impedances

function Zin = find_Zin(ZL, Zc, gamma, l)
    Zin = Zc .* ((ZL + 1i * Zc .* tan(gamma * l)) ./ (Zc + 1i * ZL .* tan(gamma * l)));
end

function Zc = find_Zc(f, R, L, G, C)
    w = 2 * pi * f;
    Zc = sqrt((R + 1i .* w .* L) ./ (G + 1i .* w .* C));
end

function [Z0o_off, Z0o_on, Z0e_off, Z0e_on, gamma_o_off, gamma_o_on, gamma_e_off, gamma_e_on, theta_o_off, theta_o_on, theta_e_off, theta_e_on] = Z0_even_odd_on_off(f, L_odd, L_even, Cm, Ca, Cst, Gm, Ga, R_odd, R_even)
    % "Off" state calculation
    [Lo_off, Co_off, Go_off, Ro_off, gamma_o_off, Z0o_off, theta_o_off] = odd_mode_params(f, L_odd, Cm, Ca, Gm, Ga, R_odd);
    [Le_off, Ce_off, Ge_off, Re_off, gamma_e_off, Z0e_off, theta_e_off] = even_mode_params(f, L_even, Cm, Ca, Gm, Ga, R_odd);
    
    % "On" state calculations
    [Lo_on, Co_on, Go_on, Ro_on, gamma_o_on, Z0o_on, theta_o_on] = odd_mode_params(f, L_odd, Cm + Cst, Ca, Gm, Ga, R_even);
    [Le_on, Ce_on, Ge_on, Re_on, gamma_e_on, Z0e_on, theta_e_on] = even_mode_params(f, L_even, Cm, Ca, Gm, Ga, R_even);


end

function [Zin_o_off, Zin_o_on, Zin_e_off, Zin_e_on] = find_all_Zins(Z0o_off, Z0o_on, Z0e_off, Z0e_on, gamma_o_off, gamma_o_on, gamma_e_off, gamma_e_on, l, ZL)

    Zin_o_off = find_Zin(ZL, Z0o_off, gamma_o_off, l);
    Zin_o_on = find_Zin(ZL, Z0o_on, gamma_o_on, l);
    Zin_e_off = find_Zin(ZL, Z0e_off, gamma_e_off, l);
    Zin_e_on = find_Zin(ZL, Z0e_on, gamma_e_on, l);

end


function [Z01, Z02, Z03, Z04] = find_Zin_for_curr_section(num_of_sections, index, Z0_curr, gamma, l, f)
    Z01 = 50 * ones(size(f));
    Z03 = Z01;

    for i = 1:index
        Z01 = find_Zin(Z01, Z0_curr, gamma, l);
    end

    for i = num_of_sections:-1:(index + 1)
        Z03 = find_Zin(Z03, Z0_curr, gamma, l);
    end

    Z02 = Z01;
    Z04 = Z03;

end

%% Even and Odd ABCD Matrices

function [Ao, Ae] = create_A_matrix(gamma0_o, gamma0_e, Z0_odd, Z0_even, l)
    Ao = zeros(2, 2, length(gamma0_o));
    Ae = Ao;

    Ao(1, 1, :) = cos(gamma0_o.*l);
    Ao(1, 2, :) = 1j.*Z0_odd.*sin(gamma0_o.*l);
    Ao(2, 1, :) = (1j./Z0_odd).*sin(gamma0_o.*l);
    Ao(2, 2, :) = cos(gamma0_o.*l);

    Ae(1, 1, :) = cos(gamma0_e.*l);
    Ae(1, 2, :) = 1j.*Z0_even.*sin(gamma0_e.*l);
    Ae(2, 1, :) = (1j./Z0_even).*sin(gamma0_e.*l);
    Ae(2, 2, :) = cos(gamma0_e.*l);

end

function [Ao_off, Ae_off, Ao_on, Ae_on] = create_on_off_A_matrices(f, L_odd, L_even, Cm, Ca, Cst, Gm, Ga, R_odd, R_even, l)
    % Compute odd mode parameters for the "off" state
    [Lo_off, Co_off, Go_off, Ro_off, gamma_o_off, Z0o_off, theta_o_off] = odd_mode_params(f, L_odd, Cm, Ca, Gm, Ga, R_odd);
    [Le_off, Ce_off, Ge_off, Re_off, gamma_e_off, Z0e_off, theta_e_off] = even_mode_params(f, L_even, Cm, Ca, Gm, Ga, R_odd);
    
    % Compute odd mode parameters for the "on" state
    [Lo_on, Co_on, Go_on, Ro_on, gamma_o_on, Z0o_on, theta_o_on] = odd_mode_params(f, L_odd, Cm + Cst, Ca, Gm, Ga, R_even);
    [Le_on, Ce_on, Ge_on, Re_on, gamma_e_on, Z0e_on, theta_e_on] = even_mode_params(f, L_even, Cm, Ca, Gm, Ga, R_even);

    [Z0o_off, Z0o_on, Z0e_off, Z0e_on, gamma_o_off, gamma_o_on, gamma_e_off, gamma_e_on, ... 
    theta_o_off, theta_o_on, theta_e_off, theta_e_on] ... 
    = Z0_even_odd_on_off(f, L_odd, L_even, Cm, Ca, Cst, Gm, Ga, R_odd, R_even);

    % Create Z matrices for "off" and "on" states (assuming create_Z_matrix is a valid function)
    [Ao_off, Ae_off] = create_A_matrix(gamma_o_off, gamma_e_off, Z0o_off, Z0e_off, l);
    [Ao_on, Ae_on] = create_A_matrix(gamma_o_on, gamma_e_on, Z0o_on, Z0e_on, l);
end

function S = a2sw(Ao, Ae, Z01o, Z01e, Z02o, Z02e)
       % create An, Bn, Cn, Dn for odd
       Ao11 = reshape(Ao(1, 1, :), [1, length(Z01o)]);
       Ao12 = reshape(Ao(1, 2, :), [1, length(Z01o)]);
       Ao21 = reshape(Ao(2, 1, :), [1, length(Z01o)]);
       Ao22 = reshape(Ao(2, 2, :), [1, length(Z01o)]);

       An_odd = (Ao11.*Z02o)./sqrt(real(Z01o) .* real(Z02o));
       Bn_odd = Ao12./sqrt(real(Z01o) .* real(Z02o));
       Cn_odd = (Ao21.*Z01o.*Z02o)./sqrt(real(Z01o) .* real(Z02o));
       Dn_odd = (Ao22.*Z01o)./sqrt(real(Z01o) .* real(Z02o));

       % create An, Bn, Cn, Dn for even
       Ae11 = reshape(Ae(1, 1, :), [1, length(Z01o)]);
       Ae12 = reshape(Ae(1, 2, :), [1, length(Z01o)]);
       Ae21 = reshape(Ae(2, 1, :), [1, length(Z01o)]);
       Ae22 = reshape(Ae(2, 2, :), [1, length(Z01o)]);
    
       An_even = (Ae11.*Z02e)./sqrt(real(Z01e) .* real(Z02e));
       Bn_even = Ae12./sqrt(real(Z01e) .* real(Z02e));
       Cn_even = (Ae21.*Z01e.*Z02e)./sqrt(real(Z01e) .* real(Z02e));
       Dn_even = (Ae22.*Z01e)./sqrt(real(Z01e) .* real(Z02e));

       S_odd = zeros(2, 2, length(Z01o));
       S_even = S_odd;

       S_odd(1, 1, :) = (An_odd + Bn_odd - Cn_odd.*((conj(Z01o))./(Z01o)) - Dn_odd.*((conj(Z01o))./(Z01o)))./(An_odd + Bn_odd + Cn_odd + Dn_odd);
       S_odd(1, 2, :) = 2.*(Ao11.*Ao22 - Ao12.*Ao21)./(An_odd + Bn_odd + Cn_odd + Dn_odd);
       S_odd(2, 1, :) = 2./(An_odd + Bn_odd + Cn_odd + Dn_odd);
       S_odd(2, 2, :) = (-An_odd.*((conj(Z02o))./(Z02o)) + Bn_odd - Cn_odd.*((conj(Z02o))./(Z02o)) + Dn_odd)./(An_odd + Bn_odd + Cn_odd + Dn_odd);

       S_even(1, 1, :) = (An_even + Bn_even - Cn_even.*((conj(Z01e))./(Z01e)) - Dn_even.*((conj(Z01e))./(Z01e)))./(An_even + Bn_even + Cn_even + Dn_even);
       S_even(1, 2, :) = 2.*(Ae11.*Ae22 - Ae12.*Ae21)./(An_even + Bn_even + Cn_even + Dn_even);
       S_even(2, 1, :) = 2./(An_even + Bn_even + Cn_even + Dn_even);
       S_even(2, 2, :) = (-An_even.*((conj(Z02e))./(Z02e)) + Bn_even - Cn_even.*((conj(Z02e))./(Z02e)) + Dn_even)./(An_even + Bn_even + Cn_even + Dn_even);

       S_A = (S_odd + S_even)/2;
       S_B = (S_odd - S_even)/2;

       S = cat(1, cat(2, S_A, S_B), cat(2, S_B, S_A));

end


%% Extracting eps(w)

function [eps_odd, eps_even] = eps_w_from_S_parameters(f, l, num_of_sections, S)
    
    S_A = S(1:2, 1:2, :);
    S_B = S(3:4, 3:4, :);
    
    S_odd = S_A + S_B;
    S_even = S_A - S_B;

    ABCD_odd = zeros(2, 2, length(f));
    ABCD_even = ABCD_odd;

    for i = 1:1:length(f)
        ABCD_odd(:, :, i) = s2abcd(S_odd(:, :, i), 50);
    end

    A_odd = reshape(ABCD_odd(1, 1, :), [1, length(f)]);
    D_odd = reshape(ABCD_odd(2, 2, :), [1, length(f)]);
    
    Beta_odd = imag( (1/(l*num_of_sections)) * acosh((A_odd + D_odd)/2) );

    eps_odd = ((Beta_odd.*9e18)./(2.*pi.*f)).^2;

    for i = 1:1:length(f)
        ABCD_even(:, :, i) = s2abcd(S_even(:, :, i), 50);
    end

    A_even = reshape(ABCD_even(1, 1, :), [1, length(f)]);
    D_even = reshape(ABCD_even(2, 2, :), [1, length(f)]);
    
    Beta_even = imag( (1/(l*num_of_sections)) * acosh((A_even + D_even)/2) );

    eps_even = ((Beta_even.*9e18)./(2.*pi.*f)).^2;
end


%% Time dependence - eps(f, t)

function plot_eps_ft(f, t, S, eps_odd, eps_even)

    S11 = reshape(S(1, 1, :), [1, length(f)]);

    eps_odd_ft = t.*(eps_odd + (2 * 1j * (2 * pi * f) .* S11) ./ (1 - S11.^2));
    eps_even_ft = t.*(eps_even + (2 * 1j * (2 * pi * f) .* S11) ./ (1 - S11.^2));

    % Create a figure with 4 subplots
    figure;

    % Plot real part of eps_odd_ft
    subplot(2, 2, 1);
    s = surf(log10(f), t, real(eps_odd_ft), 'FaceAlpha', 0.5);
    title('Real part of eps\_odd\_ft');
    xlabel('Frequency (f)');
    ylabel('Time (t)');
    s.EdgeColor = 'none';

    % Plot imaginary part of eps_odd_ft
    subplot(2, 2, 2);
    s = surf(log10(f), t, imag(eps_odd_ft), 'FaceAlpha', 0.5);
    title('Imaginary part of eps\_odd\_ft');
    xlabel('Frequency (f)');
    ylabel('Time (t)');
    s.EdgeColor = 'none';

    % Plot real part of eps_even_ft
    subplot(2, 2, 3);
    s = surf(log10(f), t, real(eps_even_ft), 'FaceAlpha', 0.5);
    title('Real part of eps\_even\_ft');
    xlabel('Frequency (f)');
    ylabel('Time (t)');
    s.EdgeColor = 'none';

    % Plot imaginary part of eps_even_ft
    subplot(2, 2, 4);
    s = surf(log10(f), t, imag(eps_even_ft), 'FaceAlpha', 0.5);
    title('Imaginary part of eps\_even\_ft');
    xlabel('Frequency (f)');
    ylabel('Time (t)');
    s.EdgeColor = 'none';

    set(gcf, 'Position', [100, 100, 1000, 1000]);

end






%% Plotting

function plot_full_S_params(f, S, title)
    gridSize = ceil(sqrt(16));
    fig_S_params = figure;
    for i = 1:16
        subplot(gridSize, gridSize, i);
        semilogx(f, reshape( 20*log10(abs(S(floor((i-1)/4)+1, mod(i-1, 4)+1, :))) , [1, length(f)] ) );
        grid on;
    end
    sgtitle(title, 'FontSize', 16);
    set(gcf, 'Position', [100, 100, 1000, 1000]);
end

function plot_S_params(f, S, title)

    gridSize = ceil(sqrt(4));
    fig_S_params = figure;
    for i = 1:4
        subplot(gridSize, gridSize, i);
        semilogx(f, reshape( 20*log10(abs(S(floor((i-1)/2)+1, mod(i-1, 2)+1, :))) , [1, length(f)] ));
        grid on;
    end
    sgtitle(title, 'FontSize', 16);
    set(gcf, 'Position', [100, 100, 1000, 1000]);
end

function plot_all_S_params(f, S_cell_array, jump)
    gridSize = ceil(sqrt(16));
    fig_S_all_params = figure('WindowStyle', 'docked');
    legends = cell(1, length(S_cell_array));

    for j = 1:jump:length(S_cell_array)
        S = S_cell_array{j};
        for i = 1:16
            subplot(gridSize, gridSize, i);
            semilogx(f, reshape(20*log10(abs(S(mod(i-1, 4)+1, floor((i-1)/4)+1, :))), [1, length(f)]));
            grid on;
            hold on;
        end
        legends{j} = ['Iteration ', num2str(j)];
    end

    sgtitle('Possible S parameters', 'FontSize', 16);
    set(gcf, 'Position', [100, 100, 1000, 1000]);
    
    % Add legend
    legend(legends);
end


function plot_full_ABCD_params(f, Ao, Ae, title)
    gridSize = ceil(sqrt(4));
    fig_A_params = figure;
    for i = 1:4
        subplot(gridSize, gridSize, i);
        semilogx(f, reshape( 20*log10(abs(Ao(floor((i-1)/2)+1, mod(i-1, 2)+1, :))) , [1, length(f)] ) );
        hold on;
        semilogx(f, reshape( 20*log10(abs(Ae(floor((i-1)/2)+1, mod(i-1, 2)+1, :))) , [1, length(f)] ) );
        grid on;
        legend('odd', 'even');
    end
    sgtitle(title, 'FontSize', 16);
    set(gcf, 'Position', [100, 100, 1000, 1000]);
end

function plot_possible_Zins(f, Zin_o_off, Zin_o_on, Zin_e_off, Zin_e_on)
    gridSize = ceil(sqrt(4));
    fig_Z_params = figure;

    subplot(gridSize, gridSize, 1);
    semilogx(f, real(Zin_o_off));
    hold on;
    semilogx(f, imag(Zin_o_off));
    grid on;

    subplot(gridSize, gridSize, 2);
    semilogx(f, real(Zin_o_on));
    hold on;
    semilogx(f, imag(Zin_o_on));
    grid on;

    subplot(gridSize, gridSize, 3);
    semilogx(f, real(Zin_e_off));
    hold on;
    semilogx(f, imag(Zin_e_off));
    grid on;

    subplot(gridSize, gridSize, 4);
    semilogx(f, real(Zin_e_on));
    hold on;
    semilogx(f, imag(Zin_e_on));
    grid on;
    

    sgtitle('Input impdeances', 'FontSize', 16);
    set(gcf, 'Position', [100, 100, 1000, 1000]);
end


%%

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
