import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()

print("data type :", type(data))

print("data shape: ", data.data.shape)

print("data keys: ", data.keys())

print("data targets name: ", data.target_names)

print("data targets shape: ", data.target.shape)

print("data featur names: ", data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Score on train data for RF: ", model.score(X_train, y_train))
print("Score on test data for RF: ", model.score(X_test, y_test))

predictions = model.predict(X_test)

number_of_test_points = len(y_test)
print("Accuacy: ", np.sum(predictions==y_test) / number_of_test_points)

# We try a simple deep learning NN

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model = MLPClassifier(max_iter=500)
model.fit(X_train_scaled, y_train)

print("Score on train data for NN: ", model.score(X_train_scaled, y_train))
print("Score on test data for NN: ", model.score(X_test_scaled, y_test))