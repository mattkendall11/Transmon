from test import egtrans, return_differences
import numpy as np
from scipy.linalg import eigh
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def to_minimize(params):
    EJt, ECt, g, fc = params
    x = return_differences(EJt, ECt, 11800, 310, g, fc)
    return -x  # We minimize the negative of x to maximize x

# Initial guess for the parameter values
initial_guess = [18400, 286, 150, 7500]
bounds = [(0,40000), (0,1000), (150,400), (0, 10000)]
result = minimize(to_minimize, initial_guess, bounds = bounds)

# Extract the optimized parameter values and the maximum x
optimal_params = result.x
max_x = -result.fun  # Remember to negate the value back to get the actual maximum x

print("Optimal Parameters:", optimal_params)
print("Maximum x:", max_x)