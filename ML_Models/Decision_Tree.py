from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
# Define the features and target as numpy arrays or lists
X = [[0, 0], [1, 1], [2, 2], [3, 3]]
y = [0, 1, 2, 3]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree object and fit the model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

num1 = float(input("Input:"))
num2 = float(input("Input:"))
# Make predictions on new data
X_new = np.array([[num1,num2]])
y_pred = dt.predict(X_new)

print("Predictions:", y_pred)
