import numpy as np
import matplotlib.pyplot as plt

from test import egtrans
from scipy.linalg import eigh
from tqdm.auto import tqdm

num = 5
clevels = num
ttlevels = num
tplevels = num
EJp = 11800
ECp = 310
EJt = 18400
ECt = 286

ec, ej = ECp/1000, EJp/1000

tp = 1000*ec*egtrans(0.5, ej/ec, 15)[0][0:tplevels]

ec, ej = ECt/1000, EJt/1000

tt = 1000*ec*egtrans(0.5,ej/ec,15)[0][0:ttlevels]

M = np.zeros((tplevels*ttlevels, tplevels * ttlevels ))

for ip in range(tplevels):
    for it in range(ttlevels):
        n = ip * ttlevels  + it
        M[n,n] = tp[ip] + tt[it]

eigenvalues, eigenvectors = eigh(M)

eigenvectors, eigenvalues = eigenvectors[::-1], eigenvalues[::-1]

pairs = []

for i in range(tplevels**2):
    for j in range(tplevels**2):
        if(
            np.round(eigenvalues[i] - eigenvalues[j],5) == np.round(tp[1],5) and
            eigenvalues[j] in tt
        ):
            pairs.append((i,j))



def offdiagonal(g):
    M2 = np.copy(M)

    # Nested loops to update M2 based on the given conditions
    for ip in range(tplevels):
        for it in range(ttlevels):

            n = ip * ttlevels  + it
            for jp in range(tplevels):
                for jt in range(ttlevels):

                    m = jp * ttlevels + jt

                    # Approximation for the interaction terms
                    ME = (
                            (jp == ip) * (jt == it + 1)  *
                            np.sqrt((it + 1) / 2) * (EJt / (8 * ECt)) ** (1 / 4) * g  +
                            (jp == ip) * (jt == it - 1)  *
                            np.sqrt(it / 2) * (EJt / (8 * ECt)) ** (1 / 4) * g  +
                            (jt == it) * (jp == ip + 1)  *
                            np.sqrt((ip + 1) / 2) * (EJp / (8 * ECp)) ** (1 / 4) * g  +
                            (jt == it) * (jp == ip - 1)  *
                            np.sqrt(ip / 2) * (EJp / (8 * ECp)) ** (1 / 4) * g
                    )

                    M2[n, m] += ME


    eigenvalues_M2, eigenvectors_M2 = np.linalg.eigh(M2)
    eigenvectors_M2, eigenvalues_M2 = eigenvectors_M2[::-1], eigenvalues_M2[::-1]
    # Define the eigenvalue differences for specific transitions
    differences = []
    for i in range(num):
        if i == 0:
            diff = eigenvalues[pairs[i][0]] - eigenvalues[pairs[i][1]]
        else:
            diff = eigenvalues_M2[pairs[i][0]] - eigenvalues_M2[pairs[i][1]]
        differences.append(diff)
    return differences


labels = [rf'|00$\rangle - |10\rangle$',rf'|01$\rangle - |11\rangle$', rf'|02$\rangle - |12\rangle$', rf'|03$\rangle - |13\rangle$',
          rf'|04$\rangle - |14\rangle$', rf'|05$\rangle - |15\rangle$']
g_values = np.linspace(0,150,500)
push_vals = []
for g in tqdm(g_values):
    push_vals.append(offdiagonal(g))
push_vals = np.array(push_vals)
push_vals = push_vals.T
for i in range(5):
    plt.plot(g_values, push_vals[i]  ,label = labels[i])
plt.title('transition differences')
plt.xlabel('coupling constant,g')
plt.ylabel('MHz')
plt.legend()
plt.show()




