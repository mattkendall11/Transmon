import numpy as np
import matplotlib.pyplot as plt

'''
assume linear relationship between Ic and A
'''

def linear_Rn_A(A):
    Rn = 1/A *166071428.6 - 1000
    return Rn
plt.ylabel(fr'Resistance $\Omega$')
plt.xlabel(fr'Overlap area $nm^2$')
a = np.linspace(5000, 100000, 1000)
Rn = linear_Rn_A(a)
plt.loglog(a,Rn)
plt.grid()
plt.show()
def linear_Ic(t):
    Ic = -0.12*t +3.482
    return Ic
t = np.linspace(10,30,50)
plt.plot(t, linear_Ic(t))
plt.xlabel(fr'oxidation time (mins)')
plt.ylabel(r'$j_c \frac{kA}{cm^2}$')
plt.grid()
plt.show()
