from scipy.stats import truncnorm
from Monte_Carlo.cavity_coupling_functions import *
from Monte_Carlo.direct_coupling_functions import *
import sys

ttlevels = 5  # target transmon number of energy levels
tplevels = 5  # probe transmon number of energy levels

initial_guess = {
    'EJt': 20000,   # Junction Energy of target transmon (MHz)
    'ECt': 400,     # Capacitor Energy of target transmon (MHz)
                    # Initial target transmon frequency 7.6 MHz
    'EJp': 15000,   # Junction Energy of probe transmon (MHz)
    'ECp': 300,     # Capacitor Energy of probe transmon (Mhz)
                    # Initial probe transmon frequency 5.7 MHz
    'g': 120,       # Coupling constant   ***
}

bounds = {
    'EJt': (15000, 25000),
    'ECt': (200, 600),
    # Target transmon frequency range: 4.69 - 10.35 MHz
    'EJp': (10000, 20000),
    'ECp': (100, 400),
    # Probe transmon frequency range: 2.73 - 7.6 MHz
    'g': (100, 150),
}


def constrain(params):
    EJt, ECt, EJp, ECp, g = params
    wt = ((8 * EJt * ECt) ** 0.5 - ECt)/1000  # target transmon frequency in GHz
    #print(wt)
    if not 4 <= wt <= 8:
        return False
    wp = ((8 * EJp * ECp) ** 0.5 - ECp) / 1000  # probe transmon frequency in GHz
    #print(wp)
    if not 4 <= wp <= 8:
        return False
    if not 1 <= wt - wp <= 2:
        return False
    if not 50 <= EJp / ECp <= 100:
        return False
    if not 50 <= EJt / ECt <= 100:
        return False
    EjtEct = EJt/ECt
    EjpEcp = EJp/ECp
    if not 20<= EjtEct<= 70:
        return False
    if not 20<= EjpEcp<= 70:
        return False
    if not 100<=g<=150:
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
    search_range_decrement = 0.999
    initial_temperature = 1
    temperature_decrement = 0.999
    return initial_search_range * search_range_decrement ** step, initial_temperature * temperature_decrement ** step


def MonteCarlo(no_samples, direct_or_indirect_coupling):
    function_to_use = return_differences if direct_or_indirect_coupling else return_differences_cavity_coupling

    curr_step = initial_guess
    curr_step['cost'] = function_to_use(*initial_guess.values(), ttlevels, tplevels)
    result = curr_step
    cost_lst = [curr_step['cost']]
    for step in tqdm(range(no_samples)):
        search_range, temperature = cooling_function(step)
        while True:
            next_step = {}
            for var in ['EJt', 'ECt', 'EJp', 'ECp', 'g']:
                next_step[var] = gaussian(curr_step[var], search_range * (bounds[var][1] - bounds[var][0]), bounds[var][0], bounds[var][1])
            if constrain(next_step.values()):
                break
        next_step['cost'] = function_to_use(*next_step.values(), ttlevels, tplevels)
        if acceptance_func(curr_step['cost'] - next_step['cost'], temperature):
            curr_step = next_step
            cost_lst.append(curr_step['cost'])
            if curr_step['cost'] > result['cost']:
                result = curr_step
    # print(search_range, temperature)
    return result, cost_lst


if __name__ == '__main__':
    """
    run monte carlo simulations from command prompt using system arguments
    open cmmd prompt go to the folder storing the project with cd "path to folder"
    type: 
    python MonteCarlo.py 5000 direct
    5000 is the first argument and is no of samples
    direct is second argument and is wheter you use direct or indirect coupling
    make sure python is in the PATH variable
    """

    res, cost = MonteCarlo(int(sys.argv[1]), (sys.argv[2] == 'direct'))
    print(res)


