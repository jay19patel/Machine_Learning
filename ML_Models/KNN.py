from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Create a custom dataset
X_train = np.array([[1, 2], [2, 1], [3, 1], [3, 4], [4, 3], [5, 4]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Create a KNN object and fit the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

num1 = float(input("Input:"))
num2 = float(input("Input:"))
# Make a prediction on a new data point
X_test = np.array([[num1, num2]])
prediction = knn.predict(X_test)

print("Prediction:", prediction)
