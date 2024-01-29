from direct_coupling_functions import return_differences
import numpy as np
import matplotlib.pyplot as plt

EJp = 10000
ECp = 200

EJt = np.linspace(8000,25000,100)
ECt = np.linspace(150,400,100)

def return_freq(ej,ec):
    return ((8 * ej * ec) ** 0.5)/1000
probe_frequency = return_freq(EJp,ECp)
print(probe_frequency)
freqs = []
t_params = []
detuning_vals = []
push_vals = []

for i in range(100):
    print(i)
    for j in range(100):

        push = return_differences(EJt[i], ECt[j], EJp, ECp, 150, 5, 5)
        freqp = return_freq(EJp, ECp)
        freqt = return_freq(EJt[i], ECt[j])
        detuning = freqp - freqt
        params = [EJt[i], ECt[j]]
        freqs.append([freqp, freqt])
        push_vals.append(push)
        detuning_vals.append(detuning)
        t_params.append(params)

np.savetxt('freqs.txt',freqs)
np.savetxt('t_params.txt', t_params)
np.savetxt('detuning_vals.txt', detuning_vals)
np.savetxt('push_vals.txt', push_vals)

freqs = np.loadtxt('freqs.txt')
t_params = np.loadtxt('t_params.txt')
detuning_vals = np.loadtxt('detuning_vals.txt')
push_vals = np.loadtxt('push_vals.txt')

ind = np.argmax(push_vals)
print(max(push_vals))
print(freqs[ind], t_params[ind], detuning_vals[ind])
ejec = []
ej = []
ec = []
for x in t_params:
    ejec.append(x[0]/x[1])
    ej.append(x[0])
    ec.append(x[1])
ft = []
for f in freqs:
    ft.append(f[1])

plt.plot(ej,push_vals, '+')
plt.title(fr'Probe frequency : {probe_frequency}')
plt.xlabel('EJ')
plt.ylabel('y (MHz)')
plt.show()




