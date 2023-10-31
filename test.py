import numpy as np

def egtrans(ng, EjEc, cutoff):
    if cutoff <= 1:
        print("Error: cutoff must be greater than 1")
        return None

    # Define 'h' as a diagonal matrix
    h = 4 * (np.arange(-cutoff, cutoff + 1) - ng)**2 * np.eye(2 * cutoff + 1)

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



