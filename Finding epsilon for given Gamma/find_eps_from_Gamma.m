syms x y z
L = 1; ZL = 100; w = 10^9; mu_0 = 4*pi*10^(-7);
Gamma = 0.5*exp(-300*z); % 1/(s+1) LPF first order

%% Direct ODE Solution

syms xi(z)

L = 1; ZL = 100; w = 10^9; mu_0 = 4*pi*10^(-7);
a = -2 *  diff(Gamma) * 1/(1-(Gamma)^2);
b = 4*1j*w*sqrt(mu_0) * Gamma * 1/(1-(Gamma)^2);


ode = diff(xi, z) == -a*xi - b*xi^2; cond = xi(L) == sqrt(mu_0)/ZL;
sol(z) = dsolve(ode, cond);
sol = sol^2;
figure
ezplot(real(sol)/(8.854187817*1e-12), [0, L])
title('esp')
legend('Derived Solution', 'MATLAB Solution')

%% Derived solution

a = -2 *  diff(Gamma) * 1/(1-(Gamma)^2);
b = 4*1j*w*sqrt(mu_0) * Gamma * 1/(1-(Gamma)^2);
A = int(a, z);
I = exp(-A);
I_L = subs(I, z, L);

D = quad(I*b, z, [L, z]);

xi = I/( I_L * (ZL/sqrt(mu_0)) + D );
eps = (xi^2);
ezplot(real(eps)/(8.854187817*1e-12), [0, L])
hold on