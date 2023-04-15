import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby1, freqz, lfilter

# Set filter parameters
L = 1e-2
N_L = 100
mu_0 = np.pi * 4e-7
eps_0 = 8.85418782e-12
omega = 2 * np.pi * (10 ** 10)
Z_L = 0.7 * 1e2
vp = 0.15e8
z = np.linspace(0, L, N_L)

k = omega / vp

# Set analog filter parameters
kc = k * 1e2
ks = k * 1e3
rp = 10

# Calculate digital filter parameters
kd = kc / ks

# Design Chebyshev Type I filter coefficients
n, Wn = cheby1(N=4, rp=rp, Wn=kd)
b, a = cheby1(N=4, rp=rp, Wn=kd)

# Generate input signal
x = np.concatenate(([1], np.zeros(N_L - 1)))

# Filter the input signal
y = lfilter(b, a, x)

# Calculate frequency response
w, h = freqz(b, a, N_L, whole=True)
h_db = 20 * np.log10(abs(h))

# Plot frequency response
fig, ax1 = plt.subplots()
ax1.set_title('Chebyshev Type I Filter Frequency Response')
ax1.plot(w, h_db, 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')
ax1.set_ylim([-60, 10])
ax1.grid()
ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
ax2.plot(w, angles, 'g')
ax2.set_ylabel('Angle (radians)', color='g')
ax2.set_ylim([-np.pi, np.pi])
plt.show()

# Plot impulse response
fig, ax = plt.subplots()
ax.stem(np.arange(N_L), y)
ax.set_title('Chebyshev Type I Filter Impulse Response')
ax.set_xlabel('Samples')
ax.set_ylabel('Amplitude')
plt.show()
