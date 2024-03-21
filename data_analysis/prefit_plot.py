import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv('data/fit_results.csv')
p = np.loadtxt('data/powers.txt')
print(df.columns)
print(df['Ql'])
plt.plot(p, df['Ql'],  label = 'internal Q-factor')
plt.plot(p,df['Qi_dia_corr'],  label = 'total Q-factor')
plt.legend()
plt.xlabel('Power (dB)')
plt.ylabel('Q-factor')
plt.show()

