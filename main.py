from test import egtrans, return_differences
import numpy as np
from scipy.linalg import eigh
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

EJp = 15000
ECp = 150
EJt = 20000
ECt = 200
g = 150
fc = 7500

# anharm = return_differences(EJt, ECt, EJp, ECp, g, fc)
# print(anharm)

def to_minimize(params):
    EJt, ECt, EJp, ECp, g, fc = params
    x = return_differences(EJt, ECt, EJp, ECp, g, fc)
    return -x  # We minimize the negative of x to maximize x

# Initial guess for the parameter values
initial_guess = [20000, 200, 20000, 150, 150, 7500]

result = minimize(to_minimize, initial_guess)

# Extract the optimized parameter values and the maximum x
optimal_params = result.x
max_x = -result.fun  # Remember to negate the value back to get the actual maximum x

print("Optimal Parameters:", optimal_params)
print("Maximum x:", max_x)