% Spacial variations

L = 1e-2;
N_L = 256;
mu_0 = pi*4e-7;
eps_0 = 8.85418782e-12;
omega = 2*pi*(10^(10));
Z_L = 0.7*1e2;
vp = 0.15e8;
z = linspace(0, L, N_L);

k = omega/vp;

% we use a chebyshev's Low Pass Filter

kc = k;
ks = k*10;

[bc,ac] = cheby1(6,10,kc/(ks/2));

% take delta function as input
x = [1, zeros(1, N_L-1)];

Gamma = filter(bc, ac, x); % Gamma(z)
dGamma = gradient(Gamma); % Gamma'(z)

% find a and b
a = -2*dGamma.*(1./(1-Gamma.^2));
b = 4*1j*omega*sqrt(mu_0)*(Gamma./(1-Gamma.^2));

% Numerically integrate a from L to z
int_a = cumtrapz(a)*(L/N_L) - trapz(a)*(L/N_L);

% Calculate I(z)
I = exp(-int_a);
D = cumtrapz(I.*b)*(L/N_L) - trapz(I.*b)*(L/N_L);

% find eps(z)
epsz = ( I(N_L)*(Z_L/sqrt(mu_0))*I.^(-1) + I.^(-1).*D ).^(-2);

figure1=figure('Position', [100, 100, 1024, 1200]);
subplot(5,1,1)
plot(z, abs(Gamma))
grid("on")
xlabel("z[m]")
ylabel("\Gamma(z)")
title("| \Gamma |")

subplot(5,1,2)
plot(z, angle(Gamma))
grid("on")
xlabel("z[m]")
ylabel("\Gamma(z)")
title("rad \Gamma")

subplot(5,1,3)
%plot(w/(2*pi), 20*log10(abs(Hz)));
freqz(bc,ac)
xlabel('Normalised k_x');
ylabel('Magnitude (dB)');
ylim([-400, 100]);
grid("on")
title('Frequency Response of \Gamma');

subplot(5,1,4)
%plot(z, abs(epsz/(eps_0)))
grid("on")
xlabel("z[m]")
ylabel("\epsilon_r(z)")
title("|\epsilon_r(z)|")

subplot(5,1,5)
plot(z, angle(epsz/(eps_0)))
grid("on")
xlabel("z[m]")
ylabel("rad \epsilon_r(z)")
title("Phase \epsilon_r(z)")