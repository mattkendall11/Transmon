from direct_coupling_functions import *
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

ttlevels = 5  # target transmon number of energy levels
tplevels = 5  # probe transmon number of energy levels
EJp = 20000  # probe transmon Junction Energy (MHz)
ECp = 310  # probe transmon Capacitor Energy (MHz)
EJt = 30000  # target transmon Junction Energy (MHz)
ECt = 286  # target transmon Capacitor Energy
g = 150  # coupling constant


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

M2 = offdiagonal(g, M, tplevels, ttlevels, EJp, ECp, EJt, ECt, pairs, True)
print(M2)

# Create a heatmap
sns.heatmap(np.where(M2==0, 1e-2, M2), norm=LogNorm())

# Show the plot
plt.xlabel('Target Energy States')
plt.ylabel('Probe Energy States')
plt.title('Hamiltonian Visualisation')
plt.show()