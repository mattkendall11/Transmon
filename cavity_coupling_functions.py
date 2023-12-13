'''
adapted from the plots files provided by andreas
'''

import numpy as np
import matplotlib.pyplot as plt
from functions import egtrans
from scipy.linalg import eigh
from tqdm.auto import tqdm
'''
define variables
'''


def offdiagonal_cavity_coupling(g, M, tplevels, ttlevels, clevels, EJp, ECp, EJt, ECt, pairs, return_M2=False):
    M2 = np.copy(M)

    # Nested loops to update M2 based on the given conditions
    for ip in range(tplevels):
        for it in range(ttlevels):
            for ic in range(clevels):
                n = ip * ttlevels * clevels + it * clevels + ic
                for jp in range(tplevels):
                    for jt in range(ttlevels):
                        for jc in range(clevels):
                            m = jp * ttlevels * clevels + jt * clevels + jc

                            # Approximation for the interaction terms
                            ME = (
                                    (jp == ip) * (jt == it + 1) * (jc == ic - 1) *
                                    np.sqrt((it + 1) / 2) * (EJt / (8 * ECt)) ** (1 / 4) * g * np.sqrt(jc + 1) +
                                    (jp == ip) * (jt == it - 1) * (jc == ic + 1) *
                                    np.sqrt(it / 2) * (EJt / (8 * ECt)) ** (1 / 4) * g * np.sqrt(jc) +
                                    (jt == it) * (jp == ip + 1) * (jc == ic - 1) *
                                    np.sqrt((ip + 1) / 2) * (EJp / (8 * ECp)) ** (1 / 4) * g * np.sqrt(jc + 1) +
                                    (jt == it) * (jp == ip - 1) * (jc == ic + 1) *
                                    np.sqrt(ip / 2) * (EJp / (8 * ECp)) ** (1 / 4) * g * np.sqrt(jc)
                            )

                            M2[n, m] += ME
    if return_M2:
        return M2

    eigenvalues_M2, eigenvectors_M2 = np.linalg.eigh(M2)
    eigenvectors_M2, eigenvalues_M2 = eigenvectors_M2[::-1], eigenvalues_M2[::-1]

    # Define the eigenvalue differences for specific transitions
    differences = []

    for i in range(1, ttlevels):
        diff = eigenvalues_M2[pairs[i][0]] - eigenvalues_M2[pairs[i][1]]
        differences.append(diff)
    return differences[::-1]


def return_differences_cavity_coupling(EJt, ECt, EJp, ECp, g, ttlevels, tplevels, clevels, fc):
    # calculate energy differences to ground state for probe and target transmon states
    tp = ECp * egtrans(0.5, EJp / ECp, 15)[0][0:tplevels]
    tt = ECt * egtrans(0.5, EJt / ECt, 15)[0][0:ttlevels]

    M = np.zeros((tplevels * ttlevels * clevels, tplevels * ttlevels * clevels))

    for ip in range(tplevels):
        for it in range(ttlevels):
            for ic in range(clevels):
                n = ip * ttlevels * clevels + it * clevels + ic
                M[n, n] = tp[ip] + tt[it] + ic * fc

    eigenvalues, eigenvectors = eigh(M)

    eigenvectors, eigenvalues = eigenvectors[::-1], eigenvalues[::-1]

    pairs = []

    for i in range(tplevels * ttlevels * clevels):
        for j in range(tplevels * ttlevels * clevels):
            if (
                    np.round(eigenvalues[i] - eigenvalues[j], 5) == np.round(tp[1], 5) and
                    eigenvalues[j] in tt
            ):
                pairs.append((i, j))
    differences = offdiagonal_cavity_coupling(g, M, tplevels, ttlevels, clevels, EJp, ECp, EJt, ECt, pairs)
    return differences[0] - differences[1]


def display_cavity_coupling(EJp, ECp, EJt, ECt, g_line, tplevels, ttlevels, clevels, fc):
    # calculate energy differences to ground state for probe and target transmon states
    tp = ECp * egtrans(0.5, EJp / ECp, 15)[0][0:tplevels]
    tt = ECt * egtrans(0.5, EJt / ECt, 15)[0][0:ttlevels]

    M = np.zeros((tplevels * ttlevels * clevels, tplevels * ttlevels * clevels))

    for ip in range(tplevels):
        for it in range(ttlevels):
            for ic in range(clevels):
                n = ip * ttlevels * clevels + it * clevels + ic
                M[n, n] = tp[ip] + tt[it] + ic * fc

    eigenvalues, eigenvectors = eigh(M)

    eigenvectors, eigenvalues = eigenvectors[::-1], eigenvalues[::-1]

    pairs = []

    for i in range(tplevels * ttlevels * clevels):
        for j in range(tplevels * ttlevels * clevels):
            if (
                    np.round(eigenvalues[i] - eigenvalues[j], 5) == np.round(tp[1], 5) and
                    eigenvalues[j] in tt
            ):
                pairs.append((i, j))


    labels = [rf'|00$\rangle - |10\rangle$', rf'|01$\rangle - |11\rangle$', rf'|02$\rangle - |12\rangle$',
              rf'|03$\rangle - |13\rangle$',
              rf'|04$\rangle - |14\rangle$', rf'|05$\rangle - |15\rangle$', rf'|06$\rangle - |16\rangle$']
    g_values = np.linspace(0, 200, 100)
    push_vals = []
    for g in tqdm(g_values):
        push_vals.append(offdiagonal_cavity_coupling(g, M, tplevels, ttlevels, clevels, EJp, ECp, EJt, ECt, pairs))
    push_vals = np.array(push_vals)
    if g_line:
        import bisect
        min_val = min(push_vals[bisect.bisect_left(g_values, g_line)])
        max_val = max(push_vals[bisect.bisect_left(g_values, g_line)])
        plt.plot([g_line] * 100, np.linspace(min_val - (max_val - min_val) / 3, max_val + (max_val - min_val) / 3, 100),
                 label='g')
    push_vals = push_vals.T
    for i in range(ttlevels-1):
        plt.plot(g_values, push_vals[i], label=labels[i])
    plt.title('transition differences')
    plt.xlabel('coupling constant,g')
    plt.ylabel('MHz')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    display_cavity_coupling(15000, 300, 20000, 400, 150, 5, 5, 5, 7500)