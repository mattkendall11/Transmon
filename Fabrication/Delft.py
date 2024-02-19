import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

# plt.plot(width, Rn, '+')
# plt.xlabel('JJ width top electrode')
# plt.ylabel(fr'$R_n$ $\Omega$')
# plt.show()

# Combine features
features = np.column_stack((r, width))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, Rn, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Plotting predictions against true values
plt.scatter(y_test, predictions)
plt.plot([0, max(y_test)], [0, max(y_test)], '--', color='red')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

#11190840.635383697
