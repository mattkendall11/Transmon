'''
adapted from the plots files provided by andreas
'''

from Transmon import TransmonInterpolation
import numpy as np

fc = 7500
clevels = 7
ttlevels = 7

tplevels = 7
EJp = 15000
ECp = 150
EJt = 20000
ECt = 200

# Create TransmonInterpolation instances for probe (tp) and target (tt)
tp = TransmonInterpolation(0.5, EJp / ECp, 15)
tt = TransmonInterpolation(0.5, EJt / ECt, 15)

# Initialize the cavity-transmon Hamiltonian matrix in an empty matrix
M = np.zeros((tplevels * ttlevels * clevels, tplevels * ttlevels * clevels))

# Fill in energies on the diagonal of bare transmon-cavity states
for ip in range(tplevels):
    for it in range(ttlevels):
        for ic in range(clevels):
            n = ip * ttlevels * clevels + it * clevels + ic
            M[n, n] = tp.energy_interp(ip, EJp / ECp) + tt.energy_interp(it, EJt / ECt) + ic * fc

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(M)

# Define the pair function to identify interested elements
def pair_function(i, j):
    return np.isclose(eigenvalues[i] - eigenvalues[j], tp.energy_interp(1, EJp / ECp)) and np.isin(eigenvalues[j], tt.energy[0])

# Use pair function to select pairs
pairs = [(i, j) for i in range(tplevels * ttlevels * clevels) for j in range(tplevels * ttlevels * clevels) if pair_function(i, j)]

# Define the interaction terms for the off-diagonal elements
ME = np.zeros_like(M)
g = 1 # Define your g value here
for ip in range(tplevels):
    for it in range(ttlevels):
        for ic in range(clevels):
            n = ip * ttlevels * clevels + it * clevels + ic
            for jp in range(tplevels):
                for jt in range(ttlevels):
                    for jc in range(clevels):
                        m = jp * ttlevels * clevels + jt * clevels + jc
                        if jp == ip and jt == it + 1 and jc == ic - 1:
                            ME[n, m] = np.sqrt((it + 1) / 2) * (EJt / (8 * ECt)) ** 0.25 * g * np.sqrt(jc + 1)
                        elif jp == ip and jt == it - 1 and jc == ic + 1:
                            ME[n, m] = np.sqrt(it / 2) * (EJt / (8 * ECt)) ** 0.25 * g * np.sqrt(jc)
                        elif jp == ip + 1 and jt == it and jc == ic - 1:
                            ME[n, m] = np.sqrt((ip + 1) / 2) * (EJp / (8 * ECp)) ** 0.25 * g * np.sqrt(jc + 1)
                        elif jp == ip - 1 and jt == it and jc == ic + 1:
                            ME[n, m] = np.sqrt(ip / 2) * (EJp / (8 * ECp)) ** 0.25 * g * np.sqrt(jc)

M2 = M + ME

# Plot the transitions of interest
import matplotlib.pyplot as plt

indices = [tt.energy[0].index(eigenvalues[pair[1]]) for pair in pairs]
colors = ['Green', 'Brown', 'Red', 'Gray', 'Blue', 'Orange', 'Purple', 'LightBlue', 'Yellow', 'Black', 'LightPurple', 'Pink', 'LightBrown', 'LightBrown', 'LightGreen', 'LightGray']

plt.figure(figsize=(12, 8))
for i, idx in enumerate(indices):
    plt.plot(g, eigenvalues[pairs[i][0]] - eigenvalues[pairs[i][1]], label=f"|000> -> |{i}00>", color=colors[i])

plt.xlabel("Coupling g strength (MHz)", fontsize=25)
plt.ylabel("Probe transition |0> -> |1> (MHz)", fontsize=25)
plt.title(f"Transition shifts of the probe in a {tplevels} lvl probe, {ttlevels} lvl target, {clevels} lvl cavity |P,T,C> Interaction system for difference state of the target", fontsize=25)
plt.legend()
plt.grid(True)
plt.show()
