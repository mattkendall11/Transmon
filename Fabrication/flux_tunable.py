import numpy as np
import matplotlib.pyplot as plt

EJ1 = 0
EJ2 = 0

def Ej(phi, EJ1 = EJ1, EJ2 = EJ2):
    a = EJ1/EJ2
    d = (a-1)/(a+1)
    sum_ej = EJ1+EJ2
    phi0 = 2.067833848*10**-15
    EJ = sum_ej*np.cos(np.pi*phi/phi0)*np.sqrt(1+d*d*(np.tan(np.pi*phi/phi0))**2)


