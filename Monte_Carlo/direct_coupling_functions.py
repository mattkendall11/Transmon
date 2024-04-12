import matplotlib.pyplot as plt
from Monte_Carlo.functions import *
from scipy.linalg import eigh
from tqdm.auto import tqdm


def offdiagonal(g, M, tplevels, ttlevels, EJp, ECp, EJt, ECt, pairs, return_M2=False):
    M2 = np.copy(M)

    # Nested loops to update M2 based on the given conditions
    for ip in range(tplevels):
        for it in range(ttlevels):

            n = ip * ttlevels + it
            for jp in range(tplevels):
                for jt in range(ttlevels):
                    m = jp * ttlevels + jt

                    # Approximation for the interaction terms
                    ME = (
                            (jp == ip) * (jt == it + 1) *
                            np.sqrt((it + 1) / 2) * (EJt / (8 * ECt)) ** (1 / 4) * g * np.sqrt(jp + 1) * (
                                    EJp / (8 * ECp)) ** (1 / 4) +
                            (jp == ip) * (jt == it - 1) *
                            np.sqrt(it / 2) * (EJt / (8 * ECt)) ** (1 / 4) * g * np.sqrt(jp) * (EJp / (8 * ECp)) ** (
                                    1 / 4) +
                            (jt == it) * (jp == ip + 1) *
                            np.sqrt((ip + 1) / 2) * (EJp / (8 * ECp)) ** (1 / 4) * g * np.sqrt(jt + 1) * (
                                    EJt / (8 * ECt)) ** (1 / 4) +
                            (jt == it) * (jp == ip - 1) *
                            np.sqrt(ip / 2) * (EJp / (8 * ECp)) ** (1 / 4) * g * np.sqrt(jt) * (EJp / (8 * ECp)) ** (
                                    1 / 4)
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


def return_differences(EJt, ECt, EJp, ECp, g, ttlevels, tplevels, ng=0.5):
    # calculate energy differences to ground state for probe and target transmon states
    tp = ECp * egtrans(ng, EJp / ECp, 15)[0][0:tplevels]
    tt = ECt * egtrans(ng, EJt / ECt, 15)[0][0:ttlevels]

    M = np.zeros((tplevels * ttlevels, tplevels * ttlevels))

    for ip in range(tplevels):
        for it in range(ttlevels):
            n = ip * ttlevels + it
            M[n, n] = tp[ip] + tt[it]

    eigenvalues, eigenvectors = eigh(M)

    eigenvectors, eigenvalues = eigenvectors[::-1], eigenvalues[::-1]

    pairs = []

    for i in range(tplevels * ttlevels):
        for j in range(tplevels * ttlevels):
            if (
                    np.round(eigenvalues[i] - eigenvalues[j], 5) == np.round(tp[1], 5) and
                    eigenvalues[j] in tt
            ):
                pairs.append((i, j))
    differences = offdiagonal(g, M, tplevels, ttlevels, EJp, ECp, EJt, ECt, pairs)
    return differences[0] - differences[1]


def display(EJt, ECt, EJp, ECp, g_line, tplevels, ttlevels, ng=0.5):
    # calculate energy differences to ground state for probe and target transmon states
    tp = ECp * egtrans(ng, EJp / ECp, 15)[0][0:tplevels]
    tt = ECt * egtrans(ng, EJt / ECt, 15)[0][0:ttlevels]

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


    labels = [rf'|00$\rangle - |10\rangle$', rf'|01$\rangle - |11\rangle$', rf'|02$\rangle - |12\rangle$',
              rf'|03$\rangle - |13\rangle$',
              rf'|04$\rangle - |14\rangle$', rf'|05$\rangle - |15\rangle$', rf'|06$\rangle - |16\rangle$']
    g_values = np.linspace(0, 250, 300)
    push_vals = []
    for g in tqdm(g_values):
        push_vals.append(offdiagonal(g, M, tplevels, ttlevels, EJp, ECp, EJt, ECt, pairs))
    push_vals = np.array(push_vals)
    if g_line:
        import bisect
        min_val = min(push_vals[bisect.bisect_left(g_values, g_line)][:ttlevels-1])
        max_val = max(push_vals[bisect.bisect_left(g_values, g_line)][:ttlevels-1])
        plt.plot([g_line]*100, np.linspace(min_val - (max_val - min_val)/3, max_val + (max_val - min_val)/3, 100), label='g')
    push_vals = push_vals.T
    for i in range(ttlevels-1):
        plt.plot(g_values, push_vals[i], label=labels[i])
    plt.title('Transition Differences')
    plt.xlabel('coupling constant,g')
    plt.ylabel('MHz')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    """
    Copy paste the params from monte carlo here to check the g plot and the detuning.
    
    """

    params = {'EJt': 15006.894040899955, 'ECt': 229.78678377191386, 'EJp': 14264.142311233556, 'ECp': 152.56651873024626, 'g': 149.64530893352173, 'cost': 7.156001451809971}
    # params = {'EJt': 15455.748591219854, 'ECt': 220.4155644397372, 'EJp': 19999.985766746653, 'ECp': 105.3390471812376, 'g': 149.99950419873878, 'cost': 12.820092751541324}
    EJt, ECt, EJp, ECp, g, tplevels, ttlevels = params['EJt'], params['ECt'], params['EJp'], params['ECp'], params['g'], 5, 5
    # print(((8 * EJt * ECt) ** 0.5 - ECt)/1000)
    # print(((8 * EJp * ECp) ** 0.5 - ECp)/1000)
    # print(return_differences(EJt, ECt, EJp, ECp, g, tplevels, ttlevels))
    # display(EJt, ECt, EJp, ECp, g, tplevels, ttlevels)

    # print(return_differences(150, ECt, EJp, ECp, g, tplevels, ttlevels, 0.05))
    # display(150, ECt, EJp, ECp, g, tplevels, ttlevels, 0.05)


    Ej_array = np.linspace(50, 5000, 200)
    ng_array = np.linspace(0.001, 0.1, 200)
    z = np.zeros((200, 200))
    for i in tqdm(range(200)):
        for j in range(200):
            z[i][j] = return_differences(Ej_array[i], ECt, EJp, ECp, g, tplevels, ttlevels, ng_array[j])
    Ej_array, ng_array = np.meshgrid(Ej_array, ng_array)

    # Create a contour plot using magnitudes
    contour = plt.contourf(Ej_array, ng_array, z, cmap='viridis')  # You can choose a different colormap

    # Add a colorbar
    plt.colorbar(contour)
    # Add labels and a title
    plt.xlabel('Ej')
    plt.ylabel('ng')
    plt.title('Dispersive shift when pulsing Ej and ng')
    # Show the plot
    plt.show()

    contour = plt.contourf(Ej_array, ng_array, abs(z), cmap='viridis')  # You can choose a different colormap

    # Add a colorbar
    plt.colorbar(contour)
    # Add labels and a title
    plt.xlabel('Ej')
    plt.ylabel('ng')
    plt.title('Dispersive shift when pulsing Ej and ng')
    # Show the plot
    plt.show()