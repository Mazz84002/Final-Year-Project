% Define Gamma function
Gamma = @(z) 0.5 * exp(-100*z);

% Define a(z) function
a = @(z) -2 * diff(Gamma(z)) ./ (1 - Gamma(z).^2);

% Define b(z) function
w = 2*pi*10e9; % Define value of w
mu0 = pi*4e-7; % Define value of mu_0
b = @(z) 4j*w*sqrt(mu0)*Gamma(z) ./ (1 - Gamma(z).^2);

% Define length L
L = 1;

% Compute I(z) function using cumtrapz function
I = @(z) -cumtrapz(L:0.01:z,a(L:0.01:z));

% Compute inverse of I(z) function using interp1 function
z_range = L:0.01:10; % Define range of z values to interpolate on
I_z_range = I(z_range);
I_inv = @(z) interp1(I_z_range, z_range, z);

% Compute epsilon(z) function
ZL = 50; % Define value of Z_L
epsilon = @(z) (1 / (I(L)*ZL/sqrt(mu0)*I_inv(z) + I_inv(z)*trapz(L:0.01:z,I(z_range).*b(z_range)))).^2;

% Plot epsilon(z) function
z_values = L:0.01:10; % Define range of z values to compute on
epsilon_values = epsilon(z_values); % Compute corresponding epsilon values

figure;
plot(z_values, real(epsilon_values), 'LineWidth', 2); % Plot real part of epsilon(z)
xlabel('z', 'FontSize', 14);
ylabel('\epsilon', 'FontSize', 14);
title('Computed \epsilon(z)', 'FontSize', 16);
