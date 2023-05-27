%% Sptial Systnesis

% input filter


[b,a] = besself(5,1e7);
[Gamma0,w] = freqs((1e-2)*b,a,logspace(4, 10, 1e4));

%{
fc = 300*1e3;
fs = 1000*1e3;
[b,a] = cheby1(6,10,fc/(fs/2));
[Gamma0, w] = freqz(1e-2*b,a,[],fs);
freqz(b,a,[],fs)
%}

% phi

phi = ifft(Gamma0);


figure;
subplot(3, 1, 1)
semilogx(w, 20*log(abs(Gamma0))/log(10))
title("\Gamma(\omega, 0)")
ylabel("Magnitude (dB)")
grid("on")
subplot(3, 1, 2)
plot(phi)
title("\phi(y)")
grid("on")


%% Temporal Synthesis



%%
function plot_resp_logy(x, y, name) % plot functions with dB in y
    subplot(2, 1, 1)
    semilogx(x, 20*log(abs(y))/log(10))
    title(name)
    ylabel("Magnitude (dB)")
    grid("on")
    subplot(2, 1, 2)
    semilogx(x, angle(y))
    ylabel("Phase (rad)")
    xlabel("\omega(rad/s)")
    grid("on")
end

function plot_resp(x, y, name) % plot functions
    subplot(2, 1, 1)
    semilogx(x, abs(y))
    title(name)
    ylabel("Magnitude (dB)")
    grid("on")
    subplot(2, 1, 2)
    semilogx(x, angle(y))
    ylabel("Phase (rad)")
    xlabel("\omega(rad/s)")
    grid("on")
end

function plot_switch_comparison(w, Gamma0, Gamma_on, Gamma_off)
    subplot(3, 1, 1)
    semilogx(w, 20*log(abs(Gamma0))/log(10))
    ylabel("\Gamma(\omega, z=0)")
    grid("on")
    subplot(3, 1, 2)
    semilogx(w, 20*log(abs(Gamma_on))/log(10))
    ylabel("\Gamma_{on}(\omega)")
    grid("on")
    subplot(3, 1, 3)
    semilogx(w, 20*log(abs(Gamma_off))/log(10))
    ylabel("\Gamma_{off}(\omega)")
    grid("on")
    xlabel("\omega(rad/s)")
end
