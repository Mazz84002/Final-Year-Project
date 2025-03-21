clear all;
close all;
%% line constants

epsilon_0 = 8.85e-12;

fstart = 1;
fstop = 17;
fpoints = 100;
f = logspace(fstart, fstop, fpoints);

%% main

num_of_sections = 8;
num_of_combinations = 2^num_of_sections;

l = 6e-6/num_of_sections;

input_matrix = create_input_matrix(num_of_sections);

[Lo, Co, Le, Ce, Ca, Cm, Cst, R, Gm, Ga] = initialise_parameters(fpoints);
[Z_mat_on, Z_mat_off] = create_on_off_Z_matrices(f, Lo, Le, Cm, Ca, Cst, Gm, Ga, R);


[Lo, Co, Go, Ro, gamma_o, Z0o, theta_o] = odd_mode_params(f, Lo, Cm, Ca, Gm, Ga, R);
[Le, Ce, Ge, Re, gamma_e, Z0e, theta_e] = even_mode_params(f, Le, Cm, Ca, Gm, Ga, R);
[Z0o_off, Z0o_on, Z0e_off, Z0e_on, gamma_o_off, gamma_o_on, gamma_e_off, gamma_e_on, theta_o_off, theta_o_on, theta_e_off, theta_e_on] = Z0_even_odd_on_off(f, Lo, Le, Cm, Ca, Cst, Gm, Ga, R);



[Zin_o_off, Zin_o_on, Zin_e_off, Zin_e_on] = find_all_Zins(Z0o_off, Z0o_on, Z0e_off, Z0e_on, gamma_o_off, gamma_o_on, gamma_e_off, gamma_e_on, l, 50);
%plot_possible_Zins(f, Zin_o_off, Zin_o_on, Zin_e_off, Zin_e_on);


% Initialize a cell array to store the identity matrices
identity_matrices = cell(1, fpoints);
% Create identity matrices and store them in the cell array
for i = 1:fpoints
    identity_matrices{i} = eye(4);
end

% Stack the identity matrices along the third dimension
S_odd = cat(3, identity_matrices{:});
S_even = S_odd;

cut_offs = ones(2, num_of_combinations);

for index = 1:1:1
    S_odd = S_even;
    for j = 1:num_of_sections
        if (input_matrix(index, j) == 1)
            [Z01_odd, Z02_odd, Z03_odd, Z04_odd] = find_Zin_for_curr_section(num_of_sections, j, find_Zc(f, Ro, Lo, Go, Co+Cst), gamma_o_on, l, f);
            [Z01_even, Z02_even, Z03_even, Z04_even] = find_Zin_for_curr_section(num_of_sections, j, find_Zc(f, Re, Le, Ge, Ce), gamma_e_on, l, f);

            plot_Z0i(f, Z01_odd, Z02_odd, Z03_odd, Z04_odd, Z01_even, Z02_even, Z03_even, Z04_even);

            F_odd = create_F(Z01_odd, Z02_odd, Z03_odd, Z04_odd);
            G_odd = create_G(Z01_odd, Z02_odd, Z03_odd, Z04_odd);
    
            S_curr_odd = z2sw(Z_mat_on, F_odd, G_odd);
            S_odd = multiply_mat(S_odd, S_curr_odd);
    
        else
            [Z01_odd, Z02_odd, Z03_odd, Z04_odd] = find_Zin_for_curr_section(num_of_sections, j, find_Zc(f, Ro, Lo, Go, Co), gamma_o_on, l, f);
            [Z01_even, Z02_even, Z03_even, Z04_even] = find_Zin_for_curr_section(num_of_sections, j, find_Zc(f, Re, Le, Ge, Ce), gamma_e_on, l, f);

            plot_Z0i(f, Z01_odd, Z02_odd, Z03_odd, Z04_odd, Z01_even, Z02_even, Z03_even, Z04_even);
    
            F_odd = create_F(Z01_odd, Z02_odd, Z03_odd, Z04_odd);
            G_odd = create_G(Z01_odd, Z02_odd, Z03_odd, Z04_odd);
    
            S_curr_odd = z2sw(Z_mat_off, F_odd, G_odd);
            S_odd = multiply_mat(S_odd, S_curr_odd);
    
        end
    
    end

end

%plot_cutoffs(cut_offs);


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

function [C_odd, C_even] = make_C_matrices(Co, Ce, Cst)

    num_of_sections = 16;
    num_of_combinations = 2^num_of_sections;
    
    % Generate all possible combinations of a 16-bit input
    input_combinations = dec2bin(0:(num_of_combinations-1), num_of_sections) - '0';
    
    % Convert the combinations to a MATLAB matrix (array)
    input_matrix = input_combinations;

    C_odd = input_matrix*(Co+Cst) + (1-input_matrix)*Co; % capacitance values for every single state
    C_even = input_matrix*(Ce) + (1-input_matrix)*Ce; % capacitance values for every single state

end

function [Lo, Co, Le, Ce, Ca, Cm, Cst, R, Gm, Ga] = initialise_parameters(fpoints)
    % Define the parameters of the quadratic function
    a = 4.0;  % Coefficient for the quadratic term
    b = 1.0;  % Coefficient for the linear term
    c = 4.0e-7;  % Coefficient for the constant term
    x = linspace(6e-7, 4e-7, fpoints);  % You can adjust the range as needed
    L = a * x.^2 + b * x + c;
    
    k = 0.66;
    Lo = L * (1 + k);
    Le = L * (1 - k);
    
    Co = 0.6e-10;
    Ce = Co/1.2;
    
    Ca = Ce;
    Cm = Co;
    
    Cst = 10 * Ce;
    
    a = 4.0;  % Coefficient for the quadratic term
    b = 1.0;  % Coefficient for the linear term
    c = 4.0;  % Coefficient for the constant term
    x = linspace(1e1, 1e7, fpoints);  % You can adjust the range as needed
    R = a * x.^2 + b * x + c;
    
    Gm = linspace(10, 1e4, fpoints);
    Ga = linspace(10, 1e4, fpoints);


end

function [Lo, Co, Go, Ro, gamma_o, Z0o, theta_o] = odd_mode_params(f, Lo, Cm, Ca, Gm, Ga, R)
    w = 2 * pi * f;
    Co = 2 * Cm - Ca;
    Go = 2 * Gm + Ga;
    Ro = R;
    
    gamma_o = sqrt((1i .* w .* Lo + Ro) .* (1i .* w .* Co + Go));
    Z0o = sqrt((Ro + 1i .* w .* Lo) ./ (Go + 1i .* w .* Co));
    
    f0_o = 1 ./ (2 .* pi .* sqrt(Lo .* Co));
    theta_o = (pi ./ 2) .* (f ./ f0_o);
end

function [Le, Ce, Ge, Re, gamma_e, Z0e, theta_e] = even_mode_params(f, Le, Cm, Ca, Gm, Ga, R)
    w = 2 * pi * f;
    Ce = Ca;
    Ge = Ga;
    Re = R;
    
    gamma_e = sqrt((1i .* w .* Le + Re) .* (1i .* w .* Ce + Ge));
    Z0e = sqrt((Re + 1i .* w .* Le) ./ (Ge + 1i .* w .* Ce));
    
    f0_e = 1 ./ (2 .* pi .* sqrt(Le .* Ce));
    theta_e = (pi ./ 2) .* (f ./ f0_e);
end

%% Input Impedances

function Zin = find_Zin(ZL, Zc, gamma, l)
    Zin = Zc .* ((ZL + 1i * Zc .* tan(gamma * l)) ./ (Zc + 1i * ZL .* tan(gamma * l)));
end

function Zc = find_Zc(f, R, L, G, C)
    w = 2 * pi * f;
    Zc = sqrt((R + 1i .* w .* L) ./ (G + 1i .* w .* C));
end

function [Z0o_off, Z0o_on, Z0e_off, Z0e_on, gamma_o_off, gamma_o_on, gamma_e_off, gamma_e_on, theta_o_off, theta_o_on, theta_e_off, theta_e_on] = Z0_even_odd_on_off(f, Lo, Le, Cm, Ca, Cst, Gm, Ga, R)
    % "Off" state calculation
    [Lo_off, Co_off, Go_off, Ro_off, gamma_o_off, Z0o_off, theta_o_off] = odd_mode_params(f, Lo, Cm, Ca, Gm, Ga, R);
    [Le_off, Ce_off, Ge_off, Re_off, gamma_e_off, Z0e_off, theta_e_off] = odd_mode_params(f, Le, Cm, Ca, Gm, Ga, R);
    
    % "On" state calculations
    [Lo_on, Co_on, Go_on, Ro_on, gamma_o_on, Z0o_on, theta_o_on] = odd_mode_params(f, Lo, Cm + Cst, Ca, Gm, Ga, R);
    [Le_on, Ce_on, Ge_on, Re_on, gamma_e_on, Z0e_on, theta_e_on] = odd_mode_params(f, Le, Cm, Ca, Gm, Ga, R);


end

function [Zin_o_off, Zin_o_on, Zin_e_off, Zin_e_on] = find_all_Zins(Z0o_off, Z0o_on, Z0e_off, Z0e_on, gamma_o_off, gamma_o_on, gamma_e_off, gamma_e_on, l, ZL)

    Zin_o_off = find_Zin(ZL, Z0o_off, gamma_o_off, l);
    Zin_o_on = find_Zin(ZL, Z0o_on, gamma_o_on, l);
    Zin_e_off = find_Zin(ZL, Z0e_off, gamma_e_off, l);
    Zin_e_on = find_Zin(ZL, Z0e_on, gamma_e_on, l);

end

function plot_possible_Zins(f, Zin_o_off, Zin_o_on, Zin_e_off, Zin_e_on)


    gridSize = ceil(sqrt(4));
    fig_S_params = figure;

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


%% Z matrix

function Z = create_Z_matrix(Z0e, cot_theta_e, csc_theta_e, Z0o, cot_theta_o, csc_theta_o)
    Z11 = -(1j/2) * (Z0e .* cot_theta_e + Z0o .* cot_theta_o);
    Z12 = -(1j/2) * (Z0e .* cot_theta_e - Z0o .* cot_theta_o);
    Z13 = -(1j/2) * (Z0e .* csc_theta_e - Z0o .* csc_theta_o);
    Z14 = -(1j/2) * (Z0e .* csc_theta_e + Z0o .* csc_theta_o);

    Z = zeros(4, 4, length(Z11));

    Z(1, 1, :) = Z11;
    Z(2, 2, :) = Z11;
    Z(3, 3, :) = Z11;
    Z(4, 4, :) = Z11;

    Z(1, 2, :) = Z12;
    Z(2, 1, :) = Z12;
    Z(3, 4, :) = Z12;
    Z(4, 3, :) = Z12;

    Z(1, 3, :) = Z13;
    Z(3, 1, :) = Z13;
    Z(2, 4, :) = Z13;
    Z(4, 2, :) = Z13;

    Z(1, 4, :) = Z14;
    Z(4, 1, :) = Z14;
    Z(2, 3, :) = Z14;
    Z(3, 2, :) = Z14;
end


function [Z_mat_on, Z_mat_off] = create_on_off_Z_matrices(f, Lo, Le, Cm, Ca, Cst, Gm, Ga, R)
   
    % Compute odd mode parameters for the "off" state
    [Lo_off, Co_off, Go_off, Ro_off, gamma_o_off, Z0o_off, theta_o_off] = odd_mode_params(f, Lo, Cm, Ca, Gm, Ga, R);
    [Le_off, Ce_off, Ge_off, Re_off, gamma_e_off, Z0e_off, theta_e_off] = odd_mode_params(f, Le, Cm, Ca, Gm, Ga, R);
    
    % Compute odd mode parameters for the "on" state
    [Lo_on, Co_on, Go_on, Ro_on, gamma_o_on, Z0o_on, theta_o_on] = odd_mode_params(f, Lo, Cm + Cst, Ca, Gm, Ga, R);
    [Le_on, Ce_on, Ge_on, Re_on, gamma_e_on, Z0e_on, theta_e_on] = odd_mode_params(f, Le, Cm, Ca, Gm, Ga, R);
    
    % Calculate cotangents and cosecants for "off" and "on" states
    cot_theta_o_off = 1 ./ tan(theta_o_off);
    cot_theta_o_on = 1 ./ tan(theta_o_on);
    
    cot_theta_e_off = 1 ./ tan(theta_e_off);
    cot_theta_e_on = 1 ./ tan(theta_e_on);
    
    csc_theta_o_off = 1 ./ sin(theta_o_off);
    csc_theta_o_on = 1 ./ sin(theta_o_on);
    
    csc_theta_e_off = 1 ./ sin(theta_e_off);
    csc_theta_e_on = 1 ./ sin(theta_e_on);
    
    
    % Create Z matrices for "off" and "on" states (assuming create_Z_matrix is a valid function)
    Z_mat_on = create_Z_matrix(Z0e_on, cot_theta_e_on, csc_theta_e_on, Z0o_on, cot_theta_o_on, csc_theta_o_on);
    Z_mat_off = create_Z_matrix(Z0e_off, cot_theta_e_off, csc_theta_e_off, Z0o_off, cot_theta_o_off, csc_theta_o_off);


end

%% Z - S convertors

function S = z2sw(Z, F, G)
    % Calculate Z - G* and Z + G
    Z_minus_G_star = Z - conj(G);
    Z_plus_G = Z + G;
    
    % Calculate the inverse of Z + G using your custom invert_mat function
    Z_plus_G_inv = invert_mat(Z_plus_G);
    
    % Calculate F^(-1) using your custom invert_mat function
    F_inv = invert_mat(F);
    
    % Calculate S = F(Z - G*)(Z + G)^(-1)F^(-1) using your custom multiply_mat function
    S = multiply_mat(multiply_mat(F, Z_minus_G_star), multiply_mat(Z_plus_G_inv, F_inv));
end

function Z = s2zw(S, F, G)
    % Calculate the identity matrix I of the same shape as S
    [M, N, P] = size(S);
    I = complex(zeros(M, N, P), zeros(M, N, P));
    
    for i = 1:P
        I(:, :, i) = eye(M, N);
    end
    
    % Calculate (I - S)^(-1) using your custom invert_mat function
    I_minus_S_inv = invert_mat(I - S);
    
    % Calculate SG + G* using your custom multiply_mat function
    SG_plus_G_star = multiply_mat(S, G) + conj(G);
    
    % Calculate F^(-1) using your custom invert_mat function
    F_inv = invert_mat(F);
    
    % Calculate Z = F^(-1)(I - S)^(-1)(SG + G*)F using your custom multiply_mat function
    Z = multiply_mat(multiply_mat(F_inv, I_minus_S_inv), multiply_mat(SG_plus_G_star, F));
end


function F = create_F(Z01, Z02, Z03, Z04)
    F = zeros(4, 4, length(Z01), 'like', Z01);
    F(1, 1, :) = 1./(2*sqrt(Z01));
    F(2, 2, :) = 1./(2*sqrt(Z02));
    F(3, 3, :) = 1./(2*sqrt(Z03));
    F(4, 4, :) = 1./(2*sqrt(Z04));
end

function G = create_G(Z01, Z02, Z03, Z04)
    G = zeros(4, 4, length(Z01), 'like', Z01);
    G(1, 1, :) = Z01;
    G(2, 2, :) = Z02;
    G(3, 3, :) = Z03;
    G(4, 4, :) = Z04;
end

function F = create_reduced_F(f)
    % Get the length of the frequency vector f
    len_f = length(f);
    
    % Initialize F as a complex zeros matrix of size (2, 2, len_f)
    F = complex(zeros(2, 2, len_f), zeros(2, 2, len_f));
    
    % Assign values to F
    F(1, 1, :) = 1 ./ (2 * sqrt(50));
    F(2, 2, :) = 1 ./ (2 * sqrt(50));
end

function G = create_reduced_G(f)
    % Get the length of the frequency vector f
    len_f = length(f);
    
    % Initialize G as a complex zeros matrix of size (2, 2, len_f)
    G = complex(zeros(2, 2, len_f), zeros(2, 2, len_f));
    
    % Assign values to G
    G(1, 1, :) = 50;
    G(2, 2, :) = 50;
end


%% Create S matrix for 4 port networks

function Z_reduced = reduce_config1(Z)
    [M, N, P] = size(Z);
    Z_reduced = complex(zeros(2, 2, P), zeros(2, 2, P));

    Z_reduced(1, 1, :) = Z(1, 1, :) - Z(1, 2, :).*Z(2, 1, :)./Z(2, 2, :);
    Z_reduced(1, 2, :) = Z(1, 4, :) - Z(1, 2, :).*Z(2, 4, :)./Z(2, 2, :);
    Z_reduced(2, 1, :) = Z(4, 1, :) - Z(4, 2, :).*Z(2, 1, :)./Z(2, 2, :);
    Z_reduced(2, 2, :) = Z(4, 4, :) - Z(4, 2, :).*Z(2, 4, :)./Z(2, 2, :);
end

function Z_reduced = reduce_config2(Z)
    Z_reduced = Z([1, 3], [1, 3], :);
end

function Z_reduced = reduce_config3(Z)
    Z_reduced = Z([1, 4], [1, 4], :);
end

function Z_reduced = reduce_config4(Z)
    Z_reduced = Z([2, 3], [2, 3], :);
end

function S_config_1 = convert_final_S_to_2_port_S_config_1(S, f)

    ZL = 50*ones(1, length(f));

    Z = zeros(4, 4, length(f), 'like', f);
    S_config_1 = zeros(2, 2, length(f), 'like', f);
    
    for i=1:length(f)
        Z(:, :, i) = s2z(S(:, :, i), 50);
    end
    Z_lpf = reduce_config1(Z);


    for i=1:length(f)
        S_config_1(:, :, i) = z2s(Z_lpf(:, :, i), 50);
    end

    
end

function S_config_2 = convert_final_S_to_2_port_S_config_2(S, f)
    
    Z = zeros(4, 4, length(f), 'like', f);
    S_config_2 = zeros(2, 2, length(f), 'like', f);
    
    for i=1:length(f)
        Z(:, :, i) = s2z(S(:, :, i), 50);
    end
    Z_lpf = reduce_config2(Z);

    for i=1:length(f)
        Z(:, :, i) = s2z(S(:, :, i), 50);
    end

    for i=1:length(f)
        S_config_2(:, :, i) = z2s(Z_lpf(:, :, i), 50);
    end
end

function S_config_3 = convert_final_S_to_2_port_S_config_3(S, f)
    
    Z = zeros(4, 4, length(f), 'like', f);
    S_config_3 = zeros(2, 2, length(f), 'like', f);
    
    for i=1:length(f)
        Z(:, :, i) = s2z(S(:, :, i), 50);
    end
    Z_lpf = reduce_config3(Z);

    for i=1:length(f)
        Z(:, :, i) = s2z(S(:, :, i), 50);
    end

    for i=1:length(f)
        S_config_3(:, :, i) = z2s(Z_lpf(:, :, i), 50);
    end
end

%% cut-offs

function S_smooth = smooth_S(S, window_size)
    % Get the size of the S matrix
    [M, N, P] = size(S);
    
    % Initialize the smoothed matrix S_smooth with the same size as S
    S_smooth = complex(zeros(M, N, P), zeros(M, N, P));
    
    % Iterate through the first and second dimensions and apply smoothing
    for i = 1:M
        for j = 1:N
            % Extract and smooth the elements along the third dimension
            S_smooth(i, j, :) = smoothdata(S(i, j, :), 'movmean', window_size);
        end
    end
end


function [cutoff_freq_1, cutoff_freq_2] = find_cutoff_frequencies(frequencies, s11_values)

    % left side
    i = 1;
    while abs(s11_values(i)) >= 0.01
        i = i+1;
    end
    cutoff_freq_1 = frequencies(i)/1e2;

    i = length(s11_values);
    while abs(s11_values(i)) >= 0.01
        i = i-1;
    end
    cutoff_freq_2 = frequencies(i)*1e4;
end

function plot_cutoffs(cut_offs)
    figure;
    
    yyaxis left;  % Set the left y-axis for "low" data
    plot(log10(cut_offs(1, : )));
    ylabel("Low Data");
    
    yyaxis right; % Set the right y-axis for "high" data
    plot(log10(cut_offs(2, : )));
    ylabel("High Data");
    
    legend("low", "high");
end




%% plots

function plot_S_params(f, S, title)
    gridSize = ceil(sqrt(4));
    fig_S_params = figure;
    for i = 1:4
        subplot(gridSize, gridSize, i);
        semilogx(f, reshape( 20*log10(abs(S(floor((i-1)/2)+1, mod(i-1, 2)+1, :))) , [1, length(f)] ) );
        grid on;
    end
    sgtitle(title, 'FontSize', 16);
    set(gcf, 'Position', [100, 100, 1000, 1000]);
end

function plot_full_Z_params(f, Z, title)
    gridSize = ceil(sqrt(16));
    fig_Z_params = figure;
    for i = 1:16
        subplot(gridSize, gridSize, i);
        semilogx(f, reshape(real(Z(floor((i-1)/4)+1, mod(i-1, 4)+1, :)), [1, length(f)]));
        grid on;
    end
    sgtitle(title, 'FontSize', 16);
    set(gcf, 'Position', [100, 100, 1500, 1500]);
end

function plot_full_S_params(f, S, title)
    gridSize = ceil(sqrt(16));
    fig_S_params = figure;
    for i = 1:16
        subplot(gridSize, gridSize, i);
        semilogx(f, reshape( 20*log10(abs(S(floor((i-1)/4)+1, mod(i-1, 4)+1, :))) , [1, length(f)] ) );
        grid on;
    end
    sgtitle(title, 'FontSize', 16);
    set(gcf, 'Position', [100, 100, 1500, 1500]);
end

function plot_all_possible_S_params(f, S_storage, title, n)
    N = length(S_storage);
    gridSize = ceil(sqrt(4));
    fig_S_params = figure;

    % Create a list of legend entries
    legend_entries = cell(1, N/n);
    for j = 1:N/n:N
        legend_entries{end+1} = ['index=' dec2bin(j-1, 8)];
    end

    % Add a legend to each subplot
    for i = 1:4
        subplot(gridSize, gridSize, i);
        for j = 1:N/n:N
            semilogx(f, 20*log10(abs(S_storage{j}(floor((i-1)/2)+1, mod(i-1, 2)+1, :))));
            hold on;
        end
        grid on;
        hold off;
    end

    % Set a common title for all subplots
    sgtitle(title, 'FontSize', 16);
    legend(legend_entries, 'Location', 'Best');
    set(gcf, 'Position', [100, 100, 1000, 1000]);
end

function plotLeftsAndRights(lefts, rights)
    % Check if the input arrays have the same length
    if length(lefts) ~= length(rights)
        error('Input arrays must have the same length');
    end

    % Create a vector for x-axis values (assuming linear spacing)
    x = 1:length(lefts);

    % Plot lefts in blue
    plot(x, log10(lefts), 'b', 'LineWidth', 1);
    hold on; % Hold the current plot

    % Plot rights in red
    plot(x, log10(rights), 'r', 'LineWidth', 1);

    % Customize plot labels and legend
    xlabel('Combination Index');
    ylabel('Frequency (Hz)');
    title('Left and Right Frequencies vs. Combination Index');
    legend('Left Frequencies', 'Right Frequencies');

    % Display the grid
    grid on;

    % Release the current plot
    hold off;
end


function plot_Z0i(f, Z01_odd, Z02_odd, Z03_odd, Z04_odd, Z01_even, Z02_even, Z03_even, Z04_even)
    fig_S_params = figure;

    subplot(4, 2, 1);
    ylabel("Z01 odd");
    semilogx(f, real(Z01_odd));
    hold on;
    semilogx(f, imag(Z01_odd));
    grid on;

    subplot(4, 2, 2);
    ylabel("Z02 odd");
    semilogx(f, real(Z02_odd));
    hold on;
    semilogx(f, imag(Z02_odd));
    grid on;

    subplot(4, 2, 3);
    ylabel("Z03 odd");
    semilogx(f, real(Z03_odd));
    hold on;
    semilogx(f, imag(Z03_odd));
    grid on;

    subplot(4, 2, 4);
    ylabel("Z04 odd");
    semilogx(f, real(Z04_odd));
    hold on;
    semilogx(f, imag(Z04_odd));
    grid on;

    subplot(4, 2, 5);
    ylabel("Z01 even");
    semilogx(f, real(Z01_even));
    hold on;
    semilogx(f, imag(Z01_even));
    grid on;

    subplot(4, 2, 6);
    ylabel("Z02 even");
    semilogx(f, real(Z02_even));
    hold on;
    semilogx(f, imag(Z02_even));
    grid on;

    subplot(4, 2, 7);
    ylabel("Z03 even");
    semilogx(f, real(Z03_even));
    hold on;
    semilogx(f, imag(Z03_even));
    grid on;

    subplot(4, 2, 8);
    ylabel("Z04 even");
    semilogx(f, real(Z04_even));
    hold on;
    semilogx(f, imag(Z04_even));
    grid on;
    

    sgtitle('Input impdeances', 'FontSize', 16);
    set(gcf, 'Position', [100, 100, 1000, 1000]);
end