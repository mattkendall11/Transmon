import numpy as np
from scipy.special import ellipk
'''

script to estimate resonant values of resonators
'''
mu0 = 1.256e-6
eps0 = 8.85e-12
epseff = 5.5#11.7#permitivity

w, gap, nturns, lcap, lmeander, r, ltail = 30, 10, 8, 50, 490, 75, 0
'''
w = width
nturns = number of full turns
lcap = dont change
l tail = dont change
'''
l = lcap +  ltail + nturns*(lmeander + r*np.pi) + np.pi/2*r
l*=1e-6

k = w/(w+2.*gap)
kp = np.sqrt(1.-k**2)

L = mu0/4*ellipk(kp)/ellipk(k)
C = 4*eps0*epseff*ellipk(k)/ellipk(kp)
vph = 1/np.sqrt(L*C)

f = vph/(4*l)

print(f/1e9)
