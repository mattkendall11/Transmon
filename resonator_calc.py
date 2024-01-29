import numpy as np
from scipy.special import ellipk

mu0 = 1.256e-6
eps0 = 8.85e-12
epseff = 12.7/2#11.7#

w, gap, nturns, lcap, lmeander, r, ltail = 30, 15, 6, 50, 490, 60,100

l = lcap +  ltail + nturns*(lmeander + r*np.pi) + np.pi/2*r
l*=1e-6

k = w/(w+2.*gap)
kp = np.sqrt(1.-k**2)

L = mu0/4*ellipk(kp)/ellipk(k)
C = 4*eps0*epseff*ellipk(k)/ellipk(kp)
vph = 1/np.sqrt(L*C)

f = vph/(4*l)

print(f/1e9)
