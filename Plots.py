'''
adapted from the plots files provided by andreas
'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from test import egtrans
from scipy.linalg import eigh
from tqdm.auto import tqdm
'''
define variables
'''
num = 7
fc = 7500
clevels = num
ttlevels = num
tplevels = num
EJp = 15000
ECp = 150
EJt = 20000
ECt = 200

ec, ej = ECp/1000, EJp/1000

tp = 1000*ec*egtrans(0.5, ej/ec, 15)[0][0:tplevels]

ec, ej = ECt/1000, EJt/1000

tt = 1000*ec*egtrans(0.5,ej/ec,15)[0][0:ttlevels]

M = np.zeros((tplevels*ttlevels*clevels, tplevels * ttlevels *clevels))

for ip in range(tplevels):
    for it in range(ttlevels):
        for ic in range(clevels):
            n = ip * ttlevels * clevels + it *clevels + ic
            M[n,n] = tp[ip] + tt[it] + ic*fc

eigenvalues, eigenvectors = eigh(M)

eigenvectors, eigenvalues = eigenvectors[::-1], eigenvalues[::-1]

pairs = []

for i in range(tplevels*ttlevels*clevels):
    for j in range(tplevels*ttlevels*clevels):
        if(
            np.round(eigenvalues[i] - eigenvalues[j],5) ==np.round(tp[1],5) and
            eigenvalues[j] in tt
        ):
            pairs.append((i,j))



def offdiagonal(g):
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
    differences = [
        #eigenvalues[pairs[ttlevels][0]] - eigenvalues[pairs[ttlevels][1]],
        #eigenvalues_M2[pairs[ttlevels][0]] - eigenvalues_M2[pairs[ttlevels][1]],
        eigenvalues_M2[pairs[ttlevels - 1][0]] - eigenvalues_M2[pairs[ttlevels - 1][1]],
        eigenvalues_M2[pairs[ttlevels - 2][0]] - eigenvalues_M2[pairs[ttlevels - 2][1]],
        eigenvalues_M2[pairs[ttlevels - 3][0]] - eigenvalues_M2[pairs[ttlevels - 3][1]],
        eigenvalues_M2[pairs[ttlevels - 4][0]] - eigenvalues_M2[pairs[ttlevels - 4][1]],
        eigenvalues_M2[pairs[ttlevels - 5][0]] - eigenvalues_M2[pairs[ttlevels - 5][1]]
    ]
    return differences


labels = [rf'|02$\rangle - |12\rangle$', rf'|03$\rangle - |13\rangle$', rf'|04$\rangle - |14\rangle$',
          rf'|05$\rangle - |15\rangle$', rf'|06$\rangle - |16\rangle$']
g_values = np.linspace(0,150,151)
push_vals = []
for g in tqdm(g_values):
    push_vals.append(offdiagonal(g))
push_vals = np.array(push_vals)
push_vals = push_vals.T
for i in range(5):
    plt.plot(g_values, push_vals[i], label = labels[i])
plt.title('transition differences')
plt.xlabel('coupling constant,g')
plt.ylabel('MHz')
plt.legend()
plt.show()