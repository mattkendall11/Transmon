import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize


def xy(f, S21, Q, f0, delx, dely, theta):
    x = S21 / (1 + 4 * Q ** 2 * (f / f0 - 1) ** 2)
    y = - S21 * 2 * Q * (f / f0 - 1) / (1 + 4 * Q ** 2 * (f / f0 - 1) ** 2)
    x = x + delx
    y = y + dely
    xr = x * np.cos(theta) - y * np.sin(theta)
    yr = y * np.cos(theta) + x * np.sin(theta)
    x, y = xr, yr
    return x, y


def ret_mag(f, S21, Q, f0, delx, dely, theta, m, n):
    x, y = xy(f, S21, Q, f0, delx, dely, theta)
    mag = m * f + n + np.sqrt(x**2 + y**2)
    return mag


def ret_phase(f, S21, Q, f0, delx, dely, theta, m, n):
    x, y = xy(f, S21, Q, f0, delx, dely, theta)
    phase = np.arctan(y/x)
    for i in range(len(phase)-1):
        if phase[i+1] - phase[i] > 3:
            phase[i+1] -= np.pi
        elif phase[i+1] - phase[i] < -3:
            phase[i+1] += np.pi
    phase = m * f + n + phase
    return phase

# EXPERIMENT WITH THEORETICAL SHAPE OF PHASE AND MAG


# f = np.linspace(1, int(1e10) + 1, int(1e5 + 1))
#
# x, y = xy(f, -10, 1e4, 5e9+1, 1, 1, np.pi/2 * 0)
#
# plt.plot(x, y, 'r-')
# plt.show()
#
# mag = ret_mag(f, -10, 1e4, 5e9+1, 1, 1, np.pi/2 * 0, 0, 0)
#
# plt.plot(f, np.sqrt(x**2 + y**2), 'b-')
# plt.show()
#
# phase = ret_phase(f, -10, 1e4, 5e9+1, 1, 1, np.pi/2 * 0, 0, 0)
#
# plt.plot(f, phase, 'b-')
# plt.show()
#
# c=1/0

# EXTRACT DATA


path = 'data/5.36 peak/501 pts 500 band 5e6 span 200 avg/'

index = 3

freq = np.loadtxt(path + 'freqs.txt')
powers = np.loadtxt(path + 'powers.txt')
mag = np.array(pd.read_csv(path + 'mags.csv', header=None).iloc[index, :].tolist())
phase = np.array(pd.read_csv(path + 'phases.csv', header=None).iloc[index, :].tolist())

print(len(powers))

# INITIAL GUESSES

min_index = np.where(mag == np.min(mag))[0][0]
f0 = freq[min_index]
mag_gradient = (mag[-1] - mag[0]) / (freq[-1] - freq[0])
mag_offset = mag[0] - mag_gradient * freq[0]
amp_S21 = mag[min_index] - (mag[0] + mag[-1]) / 2
phase_gradient = (phase[-1] - phase[0]) / (freq[-1] - freq[0])
phase_offset = phase[0] - phase_gradient * freq[0]
Q = 1e4
delx = 3
dely = 1
theta = 0

initial_params = [amp_S21, Q, f0, delx, dely, theta, mag_gradient, mag_offset, phase_gradient, phase_offset]

# PHASE 2 PI CORRECTION

for i in range(int(len(phase)/2)+1, len(phase)):
    if phase[i] - phase[i-1] > 5:
        phase[i] -= 2 * np.pi
    elif phase[i] - phase[i-1] < -5:
        phase[i] += 2 * np.pi

for i in range(int(len(phase)/2)-1, -1, -1):
    if phase[i] - phase[i+1] > 5:
        phase[i] -= 2 * np.pi
    elif phase[i] - phase[i+1] < -5:
        phase[i] += 2 * np.pi

# MAG FIT AND CORRECTION

plt.plot(freq, mag, 'r-')
plt.show()

plt.plot(freq, ret_mag(freq, amp_S21, Q, freq[min_index], delx, dely, theta, mag_gradient, mag_offset), 'b-')
plt.show()

res = curve_fit(ret_mag, freq, mag, p0=[amp_S21, Q, freq[min_index], 0, 0, 0, mag_gradient, mag_offset])
print(res[0])

plt.plot(freq, mag, 'ro')
plt.plot(freq, ret_mag(freq, *res[0]), 'b-')
plt.show()

mag -= res[0][-2] * freq
mag -= res[0][-1]

plt.plot(freq, mag, 'go')
plt.show()

# PHASE FIT AND CORRECTION

plt.plot(freq, phase, 'yo')
plt.show()
plt.plot(freq, ret_phase(freq, amp_S21, Q, freq[min_index], delx, dely, theta, phase_gradient, phase_offset), 'b-')
plt.show()

res = curve_fit(ret_phase, freq, phase, p0=[amp_S21, Q, freq[min_index], delx, dely, theta, phase_gradient, phase_offset])
print(res[0])

plt.plot(freq, phase, 'ro')
plt.plot(freq, ret_phase(freq, *res[0]), 'b-')
plt.show()

phase -= res[0][-2] * freq
phase -= res[0][-1]

plt.plot(freq, phase, 'go')
plt.show()

# CIRCLE PLOT

x = mag * np.cos(phase)
y = mag * np.sin(phase)

plt.plot(x, y, 'k.')
plt.show()


# def residuals(params, freq, mag, phase):
#     mag_predictions = ret_mag(freq, *params[:8])
#     phase_predictions = ret_phase(freq, *params[:6], *params[-2:])
#
#     residuals1 = (mag - mag_predictions) / np.mean(mag)
#     residuals2 = (phase - phase_predictions) / np.mean(phase)
#
#     return np.sum(np.concatenate((residuals1, residuals2)))
#
#
# result = minimize(residuals, initial_params, args=(freq, mag, phase))
#
# print(result['x'])
# amp_S21, Q, f0, delx, dely, theta, mag_gradient, mag_offset, phase_gradient, phase_offset = result['x']
#
# plt.plot(freq, mag, 'ro')
# plt.plot(freq, ret_mag(freq, amp_S21, Q, f0, delx, dely, theta, mag_gradient, mag_offset), 'b-')
# plt.show()
#
# plt.plot(freq, phase, 'ro')
# plt.plot(freq, ret_phase(freq, amp_S21, Q, f0, delx, dely, theta, phase_gradient, phase_offset), 'b-')
# plt.show()