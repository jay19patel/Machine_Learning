from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
# Define the features and target as numpy arrays or lists
X = [[0, 0], [1, 1], [2, 2], [3, 3]]
y = [0, 1, 2, 3]

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

num1 = float(input("Input:"))
num2 = float(input("Input:"))
# Make predictions on new data
X_new = np.array([[num1,num2]])
y_pred = rf.predict(X_new)

print("Predictions:", y_pred)

