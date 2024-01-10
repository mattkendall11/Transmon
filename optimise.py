from direct_coupling_functions import *
from scipy.optimize import minimize
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

ttlevels = 5  # target transmon number of energy levels
tplevels = 5  # probe transmon number of energy levels
EJp = 11800   # Junction Energy of probe transmon (MHz)
ECp = 310     # Capacitor Energy of probe transmon (MHz)

initial_guess = {
    'EJt': 18400,   # Junction Energy of target transmon (MHz)
    'ECt': 286,     # Capacitor Energy of target transmon (Mhz)
    'g': 150,       # Coupling constant   ***
}

bounds = {
    'EJt': (15000, 25000),
    'ECt': (200, 600),
    'g': (100, 150),
}


def to_minimize(params):
    EJt, ECt, g = params
    x = return_differences(EJt, ECt, EJp, ECp, g, ttlevels, tplevels)
    return -x  # We minimize the negative of x to maximize x


def optimise_params():
    result = minimize(to_minimize, np.array(list(initial_guess.values())), bounds=list(bounds.values()))

    # Extract the optimized parameter values and the maximum x
    optimal_params = result.x
    max_x = -result.fun  # Remember to negate the value back to get the actual maximum x

    print("Optimal Parameters:", optimal_params)
    print("Maximum x:", max_x)
    display(EJp, ECp, *optimal_params, tplevels, ttlevels)


def create_heatmap(param1, param2):
    x = np.linspace(bounds[param1][0], bounds[param1][1], 40)
    y = np.linspace(bounds[param2][0], bounds[param2][1], 40)
    z = []

    def get_param(param, i):
        if param == param1:
            return x[i]
        elif param == param2:
            return y[i]
        else:
            return initial_guess[param]
    for i in tqdm(range(40)):
        z1 = []
        for j in range(40):
            z0 = return_differences(get_param('EJt', i), get_param('ECt', j), EJp, ECp, get_param('g', i), ttlevels, tplevels)
            z1.append(np.abs(z0))
        z.append(z1)

    z = np.array(z)
    np.savetxt('z_vals_' + param1 + '_' + param2 + '.txt', z)

    x, y = np.meshgrid(x, y)

    # Create a contour plot using magnitudes
    contour = plt.contourf(x, y, z, cmap='viridis')  # You can choose a different colormap

    # Add a colorbar
    plt.colorbar(contour)
    labels = {
        'EJt': fr'$Ej_t$ (MHz)',
        'ECt': fr'$Ec_t$ (MHz)',
        'g': 'g',
    }
    # Add labels and a title
    plt.xlabel(labels[param1])
    plt.ylabel(labels[param2])
    plt.title(r'Push, $\sum |0\rangle to |1\rangle$')
    # Show the plot
    plt.show()


def load_heatmap(param1, param2):
    x = np.linspace(bounds[param1][0], bounds[param1][1], 40)
    y = np.linspace(bounds[param2][0], bounds[param2][1], 40)

    z = np.loadtxt('z_vals_' + param1 + '_' + param2 + '.txt')

    x, y = np.meshgrid(x, y)

    # Create a contour plot using magnitudes
    contour = plt.contourf(x, y, z, cmap='viridis')  # You can choose a different colormap

    # Add a colorbar
    plt.colorbar(contour)
    labels = {
        'EJt': fr'$Ej_t$ (MHz)',
        'ECt': fr'$Ec_t$ (MHz)',
        'g': 'g',
    }
    # Add labels and a title
    plt.xlabel(labels[param1])
    plt.ylabel(labels[param2])
    plt.title(r'Push, $\sum |0\rangle to |1\rangle$')
    # Show the plot
    plt.show()

if __name__ == '__main__':
    #optimise_params()
    create_heatmap('ECt', 'g')
    #load_heatmap('ECt', 'g')
