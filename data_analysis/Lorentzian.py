import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Initial guess for the parameters (amplitude, center, width, offset)
initial_guess = [-5e+10, 5.36e+9, 5e+5, -49]


# Define the Lorentzian function
def lorentzian(x, amplitude, center, width, offset):
    return amplitude / ((x - center)**2 + (width / 2)**2) + offset


# Function to fit the Lorentzian curve to the data
def fit_lorentzian(x_data, y_data):

    # Use curve_fit to fit the Lorentzian function to the data
    parameters, covariance = curve_fit(lorentzian, x_data, y_data, p0=initial_guess)

    return parameters


# Example data (replace this with your actual data)
frequencies = np.loadtxt('freqs.txt')
magnitudes = np.loadtxt('mag.txt')

# Fit Lorentzian curve to the data
fit_params = fit_lorentzian(frequencies, magnitudes)

# Generate fitted curve using the obtained parameters
fitted_curve = lorentzian(frequencies, *fit_params)
initial_curve = lorentzian(frequencies, *initial_guess)

# Plot the original data and the fitted curve
plt.plot(frequencies, magnitudes, '.', label='Original Data')
plt.plot(frequencies, fitted_curve, label='Fitted Lorentzian Curve')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Fitting Lorentzian Curve to Data')
plt.show()
