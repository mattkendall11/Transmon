import numpy as np
import matplotlib.pyplot as plt

EJ1 = 5
EJ2 = 5
Ec = 0

def Ej(phi, EJ1 = EJ1, EJ2 = EJ2):
    a = EJ1/EJ2
    d = (a-1)/(a+1)
    sum_ej = EJ1+EJ2
    phi0 = 2.067833848*10**-15
    EJ = sum_ej*np.cos(np.pi*phi)*np.sqrt(1+d*d*(np.tan(np.pi*phi))**2)
    return np.abs(EJ)


phi_vals = np.linspace(0.01,0.6,100)
ej_vals = Ej(phi_vals)
grad = np.gradient(ej_vals, phi_vals)

plt.plot(phi_vals, grad, '+', label = fr'$\alpha = 1$')

EJ1 = 5
EJ2 = 10
Ec = 0

def Ej(phi, EJ1 = EJ1, EJ2 = EJ2):
    a = EJ1/EJ2
    d = (a-1)/(a+1)
    sum_ej = EJ1+EJ2
    phi0 = 2.067833848*10**-15
    EJ = sum_ej*np.cos(np.pi*phi)*np.sqrt(1+d*d*(np.tan(np.pi*phi))**2)
    return np.abs(EJ)


phi_vals = np.linspace(0.01,0.6,100)
ej_vals = Ej(phi_vals)
grad = np.gradient(ej_vals, phi_vals)
plt.plot(phi_vals, grad, 'o', label = fr'$\alpha = 2$')

EJ1 = 5
EJ2 = 15
Ec = 0

def Ej(phi, EJ1 = EJ1, EJ2 = EJ2):
    a = EJ1/EJ2
    d = (a-1)/(a+1)
    sum_ej = EJ1+EJ2
    phi0 = 2.067833848*10**-15
    EJ = sum_ej*np.cos(np.pi*phi)*np.sqrt(1+d*d*(np.tan(np.pi*phi))**2)
    return np.abs(EJ)


phi_vals = np.linspace(0.01,0.6,100)
ej_vals = Ej(phi_vals)
grad = np.gradient(ej_vals, phi_vals)
plt.plot(phi_vals, grad, 'o', label = fr'$\alpha = 3$')
plt.xlabel(r'$\frac{\phi}{\phi_0}$')
plt.ylabel(r'd$E_J(\frac{\phi}{\phi_0})/d\phi$')
# plt.xlabel(r'$\phi$')
# plt.ylabel(r'$E_J(\phi)$')
plt.legend()
plt.show()