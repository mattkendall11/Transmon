import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from scipy.linalg import eigh
from tqdm.auto import tqdm

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

def return_differences(EJt, ECt, EJp, ECp, g, fc):
    tplevels = 4
    ttlevels = 4
    clevels = 4

    ec, ej = ECp / 1000, EJp / 1000

    tp = 1000 * ec * egtrans(0.5, ej / ec, 15)[0][0:tplevels]

    ec, ej = ECt / 1000, EJt / 1000

    tt = 1000 * ec * egtrans(0.5, ej / ec, 15)[0][0:ttlevels]

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

    eigenvalues_M2, eigenvectors_M2 = np.linalg.eigh(M2)
    eigenvectors_M2, eigenvalues_M2 = eigenvectors_M2[::-1], eigenvalues_M2[::-1]
    # Define the eigenvalue differences for specific transitions
    differences = []

    for i in range(1,clevels+1):
        diff = eigenvalues_M2[pairs[ttlevels-i][0]] - eigenvalues_M2[pairs[ttlevels - i][1]]
        differences.append(diff)

    anharmonicity = []
    for i in range(1, len(differences)):
        d = differences[i] - differences[i-1]
        anharmonicity.append(d)
    return sum(anharmonicity)

x = np.linspace(10,1000,100)
y = np.linspace(10,1000,100)
z = []
for i in tqdm(range(100)):
    z1 = []
    for j in range(100):
        z0 = return_differences(20000, x[i], 15000, y[j],150, 7500)
        z1.append(np.abs(z0))
    z.append(z1)


z = np.array(z)
np.savetxt('z_vals_Ect_Ecp.txt', z)
z = np.loadtxt('z_vals_Ect_Ecp.txt')

#magnitudes = np.linalg.norm(z, axis=0)  # You can choose axis=1 for row-wise magnitudes

# Create x and y coordinates for the contour plot
#x = np.arange(z.shape[1])
#y = np.arange(z.shape[0])
x, y = np.meshgrid(x, y)

# Create a contour plot using magnitudes
contour = plt.contourf(x, y, z, cmap='viridis')  # You can choose a different colormap

# Add a colorbar
plt.colorbar(contour)

# Add labels and a title
plt.xlabel(fr'$Ec_t$ (MHz)')
plt.ylabel(fr'$Ec_p$ (MHz)')
plt.title(r'Sum of differences, $\sum \omega_{n+1} - \omega_n$')

# Show the plot
plt.show()