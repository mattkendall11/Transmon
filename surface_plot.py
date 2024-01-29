from functions import return_differences
import numpy as np
import matplotlib.pyplot as plt

EJp = 15455.75
ECp = 220.42

#EJt = np.linspace(8000,12000,100)
ECt = np.linspace(100,300,100)
EJt = ECt*60
EJt2 = ECt*40
test = EJt/ECt

def return_freq(ej,ec):
    return ((8 * ej * ec) ** 0.5)/1000
probe_frequency = return_freq(EJp,ECp)
print(probe_frequency)

#
#
# freqs = []
# t_params = []
# detuning_vals = []
# push_vals = []
#
# for i in range(100):
#     print(i)
#     for j in range(100):
#         freqp = return_freq(EJp, ECp)
#         freqt = return_freq(EJt[i], ECt[j])
#         push = return_differences(EJt[i], ECt[j], EJp, ECp, 150, 5, 5)/freqt
#
#         detuning = (freqp - freqt)/freqt
#         params = [EJt[i], ECt[j]]
#         freqs.append([freqp, freqt])
#         push_vals.append(push)
#         detuning_vals.append(detuning)
#         t_params.append(params)
#
#
#
#
# detuning_vals2 = []
# push_vals2 = []
#
# for i in range(100):
#     print(i)
#     for j in range(100):
#         freqp = return_freq(EJp, ECp)
#         freqt = return_freq(EJt2[i], ECt[j])
#         push = return_differences(EJt2[i], ECt[j], EJp, ECp, 150, 5, 5)/freqt
#
#         detuning = (freqp - freqt)/freqt
#         push_vals2.append(push)
#         detuning_vals2.append(detuning)
#
#
#
# plt.plot(detuning_vals, push_vals, '+', label = '60')
# plt.plot(detuning_vals2, push_vals2, '+', label = '40')
# plt.xlabel(r'$\frac{\Delta}{\omega}$')
# plt.ylabel('y')
# plt.legend()
# plt.show()