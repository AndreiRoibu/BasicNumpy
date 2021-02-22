import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# Data Source: https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
dataFrame = pd.read_csv('../large_files/airfoil_self_noise.dat', sep='\t', header=None)

print(dataFrame.head())
print(dataFrame.info())

input_data = dataFrame[[0,1,2,3,4]].values
target_data = dataFrame[5].values

X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.33)

model = LinearRegression()
model.fit(X_train, y_train)

# Here we evaluate the data

print("Train Score Linear Regression:", model.score(X_train, y_train))
print("Test Score Linear Regression:", model.score(X_test, y_test))
# predictions = model.predict(X_test)
# print("Predictions:", predictions)

print('-------------------------')

model2 = RandomForestRegressor()
model2.fit(X_train, y_train)
print("Train Score Random Forest:", model2.score(X_train, y_train))
print("Test Score Random Forest:", model2.score(X_test, y_test))

print('-------------------------')

input_scaler = StandardScaler()
X_train_scaled = input_scaler.fit_transform(X_train)
X_test_scaled = input_scaler.transform(X_test)

output_scaler = StandardScaler()
y_train_scaled = output_scaler.fit_transform(np.expand_dims(y_train, -1)).ravel()
y_test_scaled = output_scaler.fit_transform(np.expand_dims(y_test, -1)).ravel()

model3 = MLPRegressor(max_iter=500)
model3.fit(X_train_scaled, y_train_scaled)

print("Train Score MLP:", model3.score(X_train_scaled, y_train_scaled))
print("Test Score MLP:", model3.score(X_test_scaled, y_test_scaled))