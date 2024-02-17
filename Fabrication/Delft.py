import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Delft_Manhattan_raw_data.csv')

df.dropna(subset=['Conductance (2W) [uS]'], inplace=True)
df = df.loc[df['Conductance (2W) [uS]'] < 1e+10 ]
df = df.loc[df['Conductance (2W) [uS]'] > 10 ]

width = df['JJ Width (top electrode)'].tolist()
conductance = df['Conductance (2W) [uS]'].to_numpy()*10**-6
x = df['X-Coordinate'].to_numpy()
y = df['Y-Coordinate'].to_numpy()
r = np.sqrt(x*x+y*y)
Rn = 1/conductance

plt.plot(width, Rn, '+')
plt.xlabel('JJ width top electrode')
plt.ylabel(fr'$R_n$ $\Omega$')
plt.show()


