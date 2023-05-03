# Import the required libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Define the input data and output data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Create an instance of the Linear Regression model
reg = LinearRegression()

# Fit the model to the input and output data
reg.fit(X, y)

num= int(input("Input : "))
# Predict the output for a new input
X_new = np.array([[num]])
y_pred = reg.predict(X_new)

# Print the predicted output
print("Predicted output:", y_pred)
