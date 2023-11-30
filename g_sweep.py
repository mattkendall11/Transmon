import numpy as np
import matplotlib.pyplot as plt
from functions import egtrans, offdiagonal
from scipy.linalg import eigh
from tqdm.auto import tqdm


ttlevels = 5  # target transmon number of energy levels
tplevels = 5  # probe transmon number of energy levels
EJp = 20000  # probe transmon Junction Energy (MHz)
ECp = 310  # probe transmon Capacitor Energy (MHz)
EJt = 30000  # target transmon Junction Energy (MHz)
ECt = 286  # target transmon Capacitor Energy (MHz)


def display(EJp, ECp, EJt, ECt, g_line, tplevels, ttlevels):
    # calculate energy differences to ground state for probe and target transmon states
    tp = ECp * egtrans(0.5, EJp / ECp, 15)[0][0:tplevels]
    tt = ECt * egtrans(0.5, EJt / ECt, 15)[0][0:ttlevels]

    M = np.zeros((tplevels * ttlevels, tplevels * ttlevels))

    for ip in range(tplevels):
        for it in range(ttlevels):
            n = ip * ttlevels + it
            M[n, n] = tp[ip] + tt[it]

    eigenvalues, eigenvectors = eigh(M)

    eigenvectors, eigenvalues = eigenvectors[::-1], eigenvalues[::-1]

    pairs = []

    for i in range(tplevels ** 2):
        for j in range(tplevels ** 2):
            if (
                    np.round(eigenvalues[i] - eigenvalues[j], 5) == np.round(tp[1], 5) and
                    eigenvalues[j] in tt
            ):
                pairs.append((i, j))


    labels = ['bare', rf'|00$\rangle - |10\rangle$', rf'|01$\rangle - |11\rangle$', rf'|02$\rangle - |12\rangle$',
              rf'|03$\rangle - |13\rangle$',
              rf'|04$\rangle - |14\rangle$', rf'|05$\rangle - |15\rangle$']
    g_values = np.linspace(0, 300, 300)
    push_vals = []
    for g in tqdm(g_values):
        push_vals.append(offdiagonal(g, M, tplevels, ttlevels, EJp, ECp, EJt, ECt, pairs, eigenvalues))
    push_vals = np.array(push_vals)
    yrange = (min(push_vals[-1])*1.01, max(push_vals[-1])*1.01)
    push_vals = push_vals.T
    for i in range(ttlevels):
        plt.plot(g_values, push_vals[i], label=labels[i])
    if g_line:
        plt.plot([g_line]*100, np.linspace(yrange[0], yrange[1], 100), label='g')
    plt.title('transition differences')
    plt.xlabel('coupling constant,g')
    plt.ylabel('MHz')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    display(EJp, ECp, EJt, ECt, None, tplevels, ttlevels)
