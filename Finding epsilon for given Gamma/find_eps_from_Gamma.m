% Analytic Solution

syms x y z s

%% 1) LPF filer 1/(s+10)


Lf = 1/(s+10);
f = ilaplace(Lf, z);
disp('f')
disp(f)
%%

Gamma = f;
disp('Gamma')
disp(Gamma)

%% Line parameters

L = 1; ZL = 1; w = 0.2; k = 2/3; mu_0 = 4*pi*10^(-7);
%%

a = -2 *  diff(Gamma) * 1/(1-(Gamma)^2);
disp('a')
disp(a)

%%

b = 4*1j*w*sqrt(mu_0) * Gamma * 1/(1-(Gamma)^2);
disp('b')
disp(b)

%%

A = int(a, z);
I = exp(-A);
disp('I')
disp(I)

I_L = subs(I, z, L);

%%

DI = int(I*b, z);
DI_L = subs(DI, z, L);

disp('DI-DI_L')
disp(DI-DI_L)

%%

xi = I/( I_L * (ZL/sqrt(mu_0)) + (DI-DI_L) );
eps = xi^2;

disp('eps')
disp(eps)

%%

figure
fplot(eps, [0, 1])
hold on

%% Direct ODE solution from MATLAB

syms xi(z)

ode = diff(xi, z) + a*xi + b*xi^2 == 0;
cond = xi(L) == sqrt(mu_0)/ZL;

disp('ODE')
disp(ode)

xiSol(z) = dsolve(ode, cond);

disp('ODE solution')
disp(xiSol)

fplot(xiSol)

