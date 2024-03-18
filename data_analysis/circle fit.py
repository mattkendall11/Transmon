import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

path = 'data/5.36 peak/1001 pts 100 band 2.5e6 span 200 avg/'
#path = 'data/full range/'

freq = np.loadtxt(path + 'freqs.txt')
powers = np.loadtxt(path + 'powers.txt')
# mag = np.loadtxt(path + 'mag.txt')
# phase = np.loadtxt(path + 'phase.txt')

mag = pd.read_csv(path + 'mags.csv').iloc[-1, :].tolist()
phase = pd.read_csv(path + 'phases.csv').iloc[-1, :].tolist()

freq, mag, phase = [v[4:-1] for v in [freq, mag, phase]]


def sym_line(x, y):
    L = len(x)
    b = (y[-1] - y[0]) / (x[-1] - x[0])
    a = 0
    min_index = np.where(y == np.min(y))[0][0]

    if min_index < (L-1)/2:
        x = x[:2 * min_index + 1]
        y = y[:2 * min_index + 1]
    elif min_index > (L-1)/2:
        x = x[2 * min_index - L + 1:]
        y = y[2 * min_index - L + 1:]

    def sym_func(params):
        a, b = params
        return sum([abs(y[i] - y[-1 - i] - a * x[i] ** 2 - b * x[i] + a * x[-1 - i] ** 2 + b * x[-1 - i]) for i in range(min_index)])

    result = minimize(sym_func, np.array([a, b]), method='BFGS')
    return *result['x'], result['message']


a, b, message = sym_line(freq, mag)
print(message)
c = mag[0] - a * freq[0] ** 2 - b * freq[0]
plt.plot(freq, mag, 'r.')
plt.plot(freq, [a * freq[i] ** 2 + b * freq[i] + c for i in range(len(freq))], 'k-')
plt.show()

# mag = [mag[i] - a * freq[i] ** 2 - b * freq[i] - c for i in range(len(freq))]


def centring_line(x, y):
    L = len(x)
    m = (y[-1] - y[0]) / (x[-1] - x[0])
    c = y[int(L / 2)] - m * x[int(L / 2)]

    def centre_func(params):
        m, c = params
        return sum([abs(y[i] - m * x[i] - c) for i in range(L)])

    result = minimize(centre_func, np.array([m, c]), method='BFGS')
    return *result['x'], result['message']


m, c, message = centring_line(freq, phase)
print(message)

plt.plot(freq, phase, 'b.')
plt.plot(freq, [m * freq[i] + c for i in range(len(freq))], 'k-')
plt.show()

phase = [phase[i] - m * freq[i] - c for i in range(len(freq))]

plt.plot(freq, mag, 'r.')
plt.show()

plt.plot(freq, phase, 'b.')
plt.show()

x = [mag[i] * np.cos(phase[i]) for i in range(len(mag))]
y = [mag[i] * np.sin(phase[i]) for i in range(len(mag))]

plt.plot(x, y, 'k.')
plt.show()