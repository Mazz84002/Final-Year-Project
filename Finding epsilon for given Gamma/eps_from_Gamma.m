L = 1;
ZL = 1;
k = 2/3;
w = 10^9;

z = linspace(0, L, 100);

Gamma = sin(z.^2);
A = (4.*1j.*w.*sqrt(4.*pi.*10.^(-7)))./(1-(Gamma).^2);
B = -2.*(1./(1-(Gamma).^2)).*[0 diff(Gamma)];

I = integral(@(z)(exp(-A).*B), L, 0);

disp("A")
disp(A)
disp("B")
disp(B)
disp("I")
disp(I)

z = L;

Gamma_L = subs(Gamma);
A_L = subs(A);
B_L = subs(B);

I = I - subs(exp(-A).*B);

syms z

xi_L = sqrt(4*pi*10^(-7))/1;
xi = 1/( (exp(A-A_L))/(xi_L) + exp(A)*I );

eps = (xi).^2;
disp("eps")
disp(eps)

figure
fplot(eps_arr, [0, 1])




