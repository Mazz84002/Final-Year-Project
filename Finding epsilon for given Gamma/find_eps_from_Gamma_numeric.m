%% Derived formula

syms x y z
L = 1; ZL = 1; w = 0.2; mu_0 = 4*pi*10^(-7);
Gamma = 0.5*exp(-100*z); % 1/(s+1) LPF first order

a = -2 *  diff(Gamma) * 1/(1-(Gamma)^2);
b = 4*1j*w*sqrt(mu_0) * Gamma * 1/(1-(Gamma)^2);
A = int(a, z);
I = exp(-A);
I_L = subs(I, z, L);

% Since D is difficult to find using symbolic MATLAB, we approximate it
% using vpaintegral and get back an array

N=100;
z_arr = linspace(0, L, N);
D_arr = zeros(1, N);
I_arr = zeros(1, N);
for i=1:N
    D_arr(i) = vpaintegral(I*b, L, z_arr(i));
end

for i=1:N
    I_arr(i) = vpaintegral(exp(-A), L, z_arr(i));
end

xi_arr = I_arr./( I_L .* (ZL./sqrt(mu_0)) + D_arr );
eps_arr = xi_arr.^2;

figure
plot(z_arr, real(eps_arr))