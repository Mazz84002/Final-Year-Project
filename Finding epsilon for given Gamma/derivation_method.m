% Define Gamma(z)
Gamma = @(z) 0.5*exp(-100*1j*z);

% Set the parameters - just random values
ZL = 1;
w = 0.2;
mu0 = 4*pi*10^(-7);

% Define the range of z values to plot
z_vals = linspace(0, 10, 1000);

L=1;
hold on
figure
y_vals = arrayfun(@(z) solve_ode(Gamma, ZL, w, mu0, L, z), z_vals);
plot(z_vals, y_vals, 'DisplayName', ['L = ' num2str(L)]);

hold off

% Add labels and legend
xlabel('z')
ylabel('y(z)')
legend('Location', 'best')


function [y] = solve_ode(Gamma, ZL, w, mu0, L, z)
% Define functions a(z) and b(z)
a = @(z) -2*exp(-z)./(1-Gamma(z).^2);
b = @(z) 4j*w*sqrt(mu0)*Gamma(z)./(1-Gamma(z).^2);

% Define function I(z)
I = @(z) exp(-integral(@(x) a(x), -Inf, z, 'ArrayValued', true));

% Define function y(z)
integrand_L_z = integral(@(x) I(x).*b(x), L, z, 'ArrayValued', true);
I_L = I(L);
I_inv_z = 1./I(z);
I_inv_integrand_L_z = 1./(I_L*(ZL/mu0)*I_inv_z + I_inv_z*integrand_L_z);
y = I_inv_integrand_L_z.^2;
end
