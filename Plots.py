'''
adapted from the plots files provided by andreas
'''

import numpy as np
import matplotlib.pyplot as plt
from Transmon import egtrans

# Define parameters
fc = 7500
clevels = 7
ttlevels = 7
tplevels = 7
EJp = 15000
ECp = 150
EJt = 20000
ECt = 200

# Calculate energy spectrum
tp = 1000 * (egtrans(0.5, EJp / 1000.0, 15)[:, :tplevels])
tt = 1000 * (egtrans(0.5, EJt / 1000.0, 15)[:, :ttlevels])

# Initialize the Hamiltonian matrix
matrix_size = tplevels * ttlevels * clevels
M = np.zeros((matrix_size, matrix_size))

# Fill in energies on the diagonal
for ip in range(tplevels):
    for it in range(ttlevels):
        for ic in range(clevels):
            n = ip * ttlevels * clevels + it * clevels + ic
            M[n, n] = tp[ip] + tt[it] + ic * fc

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(M)

# Define a function to check if a pair meets specific conditions
def check_pair(i, j):
    return (
        eigenvalues[i] - eigenvalues[j] == tp[1]
        and eigenvalues[j] in tt
    )

# Find pairs meeting the conditions
pairs = [(i, j) for i in range(matrix_size) for j in range(matrix_size) if check_pair(i, j)]

# Initialize a new matrix
M2 = M.copy()

# Fill in interaction terms
for ip in range(tplevels):
    for it in range(ttlevels):
        for ic in range(clevels):
            n = ip * ttlevels * clevels + it * clevels + ic
            for jp in range(tplevels):
                for jt in range(ttlevels):
                    for jc in range(clevels):
                        m = jp * ttlevels * clevels + jt * clevels + jc
                        # Calculate ME
                        ME = 0  # You will need to adapt this based on the provided expression
                        M2[n, m] += ME

# Plot transitions
# You will need to adapt this part to plot the transitions in Python
