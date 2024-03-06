import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the Lorentzian function
def lorentzian(x, amplitude, center, width,c):
    return (amplitude / ((x - center)**2 + (width / 2)**2) )+c

# Function to fit the Lorentzian curve to the data
def fit_lorentzian(x_data, y_data):
    # Initial guess for the parameters (amplitude, center, width, offset)
    initial_guess = [-5e+10, 5.36e9, 5e5, -49]

    # Use curve_fit to fit the Lorentzian function to the data
    parameters, covariance = curve_fit(lorentzian, x_data, y_data, maxfev=100000, p0=initial_guess)

    return parameters

# Example data (replace this with your actual data)
x_data = np.loadtxt('freqs.txt')
y_data = np.loadtxt('mag.txt')

# Fit Lorentzian curve to the data
fit_params = fit_lorentzian(x_data, y_data)
print(fit_params)
initial_guess = [0.5e7, 5.36*10**9, 0.5*10**9, 48]
# Generate fitted curve using the obtained parameters
fitted_curve = lorentzian(x_data, *fit_params)

# Plot the original data and the fitted curve
plt.plot(x_data, y_data, 'o', label='Original Data')
plt.plot(x_data, fitted_curve, label='Fitted Lorentzian Curve')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Fitting Lorentzian Curve to Data')
plt.show()
