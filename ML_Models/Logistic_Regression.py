from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Create custom dataset
X = np.array([[0.5, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, 2.5], [2.5, 3.0], [3.0, 3.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# Create a logistic regression object
clf = LogisticRegression()

# Train the model using the data
clf.fit(X, y)
num1 = float(input("Input:"))
num2 = float(input("Input:"))
# Make predictions on new data
X_new = np.array([[num1,num2]])
y_pred = clf.predict(X_new)

print("Predictions:", y_pred)

