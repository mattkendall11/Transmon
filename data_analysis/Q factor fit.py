import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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


class Qfactor:
    def __init__(self, path):
        self.freq = np.loadtxt(path + 'freqs.txt')
        mag_csv = pd.read_csv(path + 'mags.csv', header=None)
        self.powers = np.loadtxt(path + 'powers.txt')[:len(mag_csv)]
        self.mag = [np.array(pd.read_csv(path + 'mags.csv', header=None).iloc[index, :].tolist()) for index in range(len(self.powers))]
        self.phase = [np.array(pd.read_csv(path + 'phases.csv', header=None).iloc[index, :].tolist()) for index in range(len(self.powers))]
        self.res_mag = []
        self.res_phase = []

    def fit(self, initial_guess = [1e4, 10, 3, 0]):
        Q, delx, dely, theta = initial_guess
        Qi = []
        Qi_err = []
        for index in range(len(self.powers)):
            mag = self.mag[index]
            phase = self.phase[index]

            for i in range(int(len(phase) / 2) + 1, len(phase)):
                if phase[i] - phase[i - 1] > 5:
                    phase[i] -= 2 * np.pi
                elif phase[i] - phase[i - 1] < -5:
                    phase[i] += 2 * np.pi

            for i in range(int(len(phase) / 2) - 1, -1, -1):
                if phase[i] - phase[i + 1] > 5:
                    phase[i] -= 2 * np.pi
                elif phase[i] - phase[i + 1] < -5:
                    phase[i] += 2 * np.pi

            min_index = np.where(mag == np.min(mag))[0][0]
            f0 = self.freq[min_index]
            mag_gradient = (mag[-1] - mag[0]) / (self.freq[-1] - self.freq[0])
            mag_offset = mag[0] - mag_gradient * self.freq[0]
            amp_S21 = mag[min_index] - (mag[0] + mag[-1]) / 2
            phase_gradient = (phase[-1] - phase[0]) / (self.freq[-1] - self.freq[0])
            phase_offset = phase[0] - phase_gradient * self.freq[0]

            res_mag = curve_fit(ret_mag, self.freq, mag, p0=[amp_S21, Q, f0, delx, dely, theta, mag_gradient, mag_offset], maxfev=int(1e5))

            self.res_mag.append(res_mag[0])

            res_phase = curve_fit(ret_phase, self.freq, phase, p0=[amp_S21, Q, f0, delx, dely, theta, phase_gradient, phase_offset], maxfev=int(1e5))

            self.res_phase.append(res_phase[0])

            Qi.append(res_mag[0][1])
            Qi_err.append(np.sqrt(res_mag[1][1][1]))
        plt.errorbar(self.powers, Qi, yerr=Qi_err, fmt='o', color='blue', label='Data')
        plt.show()
        plt.plot(self.powers, Qi, 'ro')
        plt.show()

    def initial_params(self, index, initial_guess):
        Q, delx, dely, theta = initial_guess
        mag = self.mag[index]
        phase = self.phase[index]

        for i in range(int(len(phase) / 2) + 1, len(phase)):
            if phase[i] - phase[i - 1] > 5:
                phase[i] -= 2 * np.pi
            elif phase[i] - phase[i - 1] < -5:
                phase[i] += 2 * np.pi

        for i in range(int(len(phase) / 2) - 1, -1, -1):
            if phase[i] - phase[i + 1] > 5:
                phase[i] -= 2 * np.pi
            elif phase[i] - phase[i + 1] < -5:
                phase[i] += 2 * np.pi

        min_index = np.where(mag == np.min(mag))[0][0]
        f0 = self.freq[min_index]
        mag_gradient = (mag[-1] - mag[0]) / (self.freq[-1] - self.freq[0])
        mag_offset = mag[0] - mag_gradient * self.freq[0]
        amp_S21 = mag[min_index] - (mag[0] + mag[-1]) / 2
        phase_gradient = (phase[-1] - phase[0]) / (self.freq[-1] - self.freq[0])
        phase_offset = phase[0] - phase_gradient * self.freq[0]

        plt.plot(self.freq, mag, 'ro')
        plt.plot(self.freq, ret_mag(self.freq, amp_S21, Q, f0, delx, dely, theta, mag_gradient, mag_offset), 'b-')
        plt.show()

        plt.plot(self.freq, phase, 'ro')
        plt.plot(self.freq, ret_phase(self.freq, amp_S21, Q, f0, delx, dely, theta, phase_gradient, phase_offset), 'b-')
        plt.show()

    def display(self, index):
        mag = self.mag[index]
        phase = self.phase[index]
        res_mag = self.res_mag[index]
        res_phase = self.res_phase[index]

        plt.plot(self.freq, mag, 'ro')
        plt.plot(self.freq, ret_mag(self.freq, *res_mag), 'b-')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Magnitude (dB)')
        plt.show()

        plt.plot(self.freq, phase, 'ro')
        plt.plot(self.freq, ret_phase(self.freq, *res_phase), 'b-')

        plt.show()

        mag -= res_mag[-2] * self.freq
        mag -= res_mag[-1]

        phase -= res_phase[-2] * self.freq
        phase -= res_phase[-1]

        x = mag * np.cos(phase)
        y = mag * np.sin(phase)

        plt.plot(x, y, 'k.')
        plt.show()


if __name__ == '__main__':
    # path = 'data/5.36 peak/251 pts 200 band 2e6 span 2000 avg/'
    # path = 'data/5.36 peak/251 pts 200 band 5e6 span 1000 avg/'
    path = 'data/5.36 peak/501 pts 100 band 2.5e6 span 200 avg/'
    # path = 'data/5.36 peak/501 pts 500 band 2e6 span 200 avg/'
    # path = 'data/5.36 peak/501 pts 500 band 5e6 span 200 avg/'
    # path = 'data/5.36 peak/501 pts 500 band 5e7 span 200 avg/'
    # path = 'data/5.36 peak/1001 pts 100 band 2.5e6 span 200 avg/'
    # path = 'data/5.36 peak/1001 pts 100 band 2.5e6 span 200 avg/'m
    # path = 'data/5.83 peak/251 pts 100 band 7e6 span 100 avg/'
    # path = 'data/5.83 peak/251 pts 500 band 1e7 span 20 avg/'
    # path = 'data/5.83 peak/501 pts 100 band 5e6 span 100 avg/'


    Q1 = Qfactor(path)
    Q1.fit([1e4, 10, 3, 0])
    # Q1.initial_params(0, [1e4, 10, 3, 0])
    Q1.display(13)