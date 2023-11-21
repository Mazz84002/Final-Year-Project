%% Sptial Systnesis

% input filter

% Define parameters
fs = 1e8;                % Sampling frequency (Hz)
fc = 1e7;                % Cutoff frequency (Hz)
ripple = 0.5;            % Passband ripple (dB)
filter_order = 6;        % Filter order

% Design Chebyshev type I low-pass filter
[b, a] = cheby1(filter_order, ripple, fc / (fs/2));

% Compute frequency response of the filter
freq = linspace(0, fs/2, 1000);
Gamma0 = freqz(b, a, freq, fs);

% Plot the filter frequency response
figure;
plot(freq, abs(Gamma0));
title('Chebyshev Type I Low-Pass Filter');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

% Compute impulse response of the filter
phi = ifft(Gamma0);

% Plot the impulse response
t = linspace(0, length(phi)/fs, length(phi));
figure;
plot(t, real(phi));
hold on;
plot(t, imag(phi));
title('Impulse Response of Chebyshev Type I Low-Pass Filter');
xlabel('Time (s)');
ylabel('Magnitude');


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
