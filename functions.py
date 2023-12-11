import numpy as np
from scipy.linalg import eigh


def egtrans(ng, EjEc, cutoff):
    """

    :param ng: offset charge (dimensionless)
    :param EjEc: Junction Energy to Capacitor Energy Ratio (dimensionless)
    :param cutoff: number of energy levels
    :return: energy differences between all states and ground state
    """

    if cutoff <= 1:
        print("Error: cutoff must be greater than 1")
        return None

    # Define 'h' as a diagonal matrix
    h = 4 * (np.arange(-cutoff, cutoff + 1) - ng) ** 2 * np.eye(2 * cutoff + 1)

    # Define 'hv' as a sparse matrix
    hv = -np.eye(2 * cutoff + 1)
    for i in range(2 * cutoff):
        hv[i, i + 1] = -1.0
        hv[i + 1, i] = -1.0

    # Define 'n' as a diagonal matrix
    n = np.diag(np.arange(-cutoff, cutoff + 1))

    # Calculate eigenvalues and eigenvectors
    e, v = np.linalg.eig(h + EjEc * hv / 2)

    # Sort eigenvalues and eigenvectors
    o = np.argsort(e)
    e2 = e[o]
    v2 = v[:, o]

    # Calculate 'g' matrix
    g = np.dot(np.dot(v2, n), v2.T)

    # Calculate sign of 'g'
    sgn = np.sign(g)
    g = sgn * g

    # Calculate 'de' and 'dv2'
    de = np.array([np.dot(v2[:, i], np.dot(hv, v2[:, i])) / 2 for i in range(2 * cutoff + 1)])
    # Calculate 'dv2' with division by zero check
    dv2 = np.zeros((2 * cutoff + 1, 2 * cutoff + 1))
    for i in range(2 * cutoff + 1):
        for j in range(2 * cutoff + 1):
            if i == j:
                dv2[i, j] = 0
            else:
                if e2[i] == e2[j]:
                    dv2[i, j] = 0
                else:
                    dv2[i, j] = sum((v2[j, i] * v2[j, i] * np.dot(hv, v2[i, :])) / (e2[i] - e2[j]))

    # Convert dv2 to a NumPy array
    dv2 = np.array(dv2)

    # Calculate 'dg' matrix
    dg = sgn * (np.dot(np.dot(dv2, n), v2.T) + np.dot(np.dot(v2, n), dv2.T)) / 2

    # Return results
    return e2 - e2[0], de - de[0], g, dg


def offdiagonal(g, M, tplevels, ttlevels, EJp, ECp, EJt, ECt, pairs, eigenvalues, return_M2=False):
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
    # for i in range(ttlevels):
    #     if i == 0:
    #         diff = eigenvalues[pairs[i][0]] - eigenvalues[pairs[i][1]]
    #     else:
    #         diff = eigenvalues_M2[pairs[i][0]] - eigenvalues_M2[pairs[i][1]]
    #     differences.append(diff)
    for i in range(ttlevels):

        diff = eigenvalues_M2[pairs[i][0]] - eigenvalues_M2[pairs[i][1]]
        differences.append(diff)
    return differences[::-1]


def return_differences(EJt, ECt, EJp, ECp, g, ttlevels, tplevels):
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
    differences = offdiagonal(g, M, tplevels, ttlevels, EJp, ECp, EJt, ECt, pairs, eigenvalues)
    return differences[0] - differences[1]



