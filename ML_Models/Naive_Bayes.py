from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np

# Create a custom dataset
X = np.array([[1, 2, 1], [2, 3, 4], [3, 1, 2], [4, 2, 5], [5, 1, 3], [6, 2, 1], [7, 3, 5], [8, 1, 2]])
y = np.array([0, 1, 0, 1, 1, 0, 1, 0])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Naive Bayes object and fit the model
nb = GaussianNB()
nb.fit(X_train, y_train)

num1 = float(input("Input:"))
num2 = float(input("Input:"))
num3 = float(input("Input:"))
# Make predictions on new data
X_new = np.array([[num1,num2,num3]])
y_pred = nb.predict(X_new)

print("Predictions:", y_pred)