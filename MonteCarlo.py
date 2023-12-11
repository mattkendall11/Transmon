import numpy as np
from functions import return_differences
from tqdm.auto import tqdm
from scipy.stats import truncnorm
from g_sweep import display


ttlevels = 5  # target transmon number of energy levels
tplevels = 5  # probe transmon number of energy levels

initial_guess = {
    'EJt': 20000,   # Junction Energy of target transmon (MHz)
    'ECt': 400,     # Capacitor Energy of target transmon (MHz)
                    # Initial target transmon frequency 5.7 MHz
    'EJp': 15000,   # Junction Energy of probe transmon (MHz)
    'ECp': 300,     # Capacitor Energy of probe transmon (Mhz)
                    # Initial probe transmon frequency 7.6 MHz
    'g': 120,       # Coupling constant   ***
}

bounds = {
    'EJt': (15000, 25000),
    'ECt': (200, 600),
    # Target transmon frequency range: 2.73 - 7.6 MHz
    'EJp': (10000, 20000),
    'ECp': (100, 400),
    # Probe transmon frequency range: 4.69 - 10.35 MHz
    'g': (100, 250),
}


def constrain(params):
    EJt, ECt, EJp, ECp, g = params
    wt = ((8 * EJt * ECt) ** 0.5 - ECt)/1000  # target transmon frequency in GHz
    if not 4 <= wt <= 8:
        return False
    wp = ((8 * EJp * ECp) ** 0.5 - ECp) / 1000  # probe transmon frequency in GHz
    if not 4 <= wp <= 8:
        return False
    if not 1 <= wp - wt <= 2:
        return False
    return True


def gaussian(mean, std, low=0, upp=1):
    return truncnorm(
        (low - mean) / std, (upp - mean) / std, loc=mean, scale=std).rvs()


def acceptance_func(delx, T):
    r = np.random.rand()
    if delx <= 0:
        return True
    return np.exp(- delx / T) > r


def cooling_function(step):
    initial_search_range = 1
    search_range_decrement = 0.99
    initial_temperature = 30
    temperature_decrement = 0.99
    return initial_search_range * search_range_decrement ** step, initial_temperature * temperature_decrement ** step


def MonteCarlo(no_samples):
    result = initial_guess
    cost = return_differences(*initial_guess.values(), ttlevels, tplevels)
    for step in tqdm(range(no_samples)):
        search_range, temperature = cooling_function(step)
        while True:
            next_step = {}
            for var in ['EJt', 'ECt', 'EJp', 'ECp', 'g']:
                next_step[var] = gaussian(result[var], search_range * (bounds[var][1] - bounds[var][0]), bounds[var][0], bounds[var][1])
            if constrain(next_step.values()):
                break
        new_cost = return_differences(*result.values(), ttlevels, tplevels)
        if acceptance_func(cost - new_cost, temperature):
            result = next_step
            cost = new_cost
    return result, cost


if __name__ == '__main__':
    result, cost = MonteCarlo(1000)
    print(result)
    print(cost)
    # print(return_differences(18404.25049977,   161.86999255, 11800, 310, 150, 5, 5))
    # display(18404.25049977,   161.86999255, 11800, 310, 150, 5, 5)
