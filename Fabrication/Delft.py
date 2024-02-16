import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Delft_Manhattan_raw_data.csv')

df.dropna(subset=['Conductance (2W) [uS]'], inplace=True)
df = df.loc[df['Conductance (2W) [uS]'] < 1e+20]
print(df)
width = df['JJ Width (top electrode)'].tolist()
conductance = df['Conductance (2W) [uS]'].tolist()
print(conductance)
print(width)
plt.plot(width, conductance, '+')
plt.xlabel('JJ width top electrode')
plt.ylabel('Conductance (2W) [uS]')
plt.show()
