import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the Lorentzian function
def lorentzian(x, amplitude, center, width):
    return amplitude / ((x - center)**2 + (width / 2)**2)

# Function to fit the Lorentzian curve to the data
def fit_lorentzian(x_data, y_data):
    # Initial guess for the parameters (amplitude, center, width)
    initial_guess = [max(y_data), np.mean(x_data), 1.0]

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

# Plot the original data and the fitted curve
plt.plot(frequencies, magnitudes, 'o', label='Original Data')
plt.plot(frequencies, magnitudes, label='Fitted Lorentzian Curve')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Fitting Lorentzian Curve to Data')
plt.show()
