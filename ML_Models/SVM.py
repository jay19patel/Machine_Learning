import numpy as np
from sklearn import svm

# Create a custom dataset
X = np.array([[0, 0], [1, 1]])
y = np.array([0, 1])

# Create an SVM object and train the model
clf = svm.SVC()
clf.fit(X, y)

num1 = float(input("Input:"))
num2 = float(input("Input:"))
# Make a prediction on a new data point
new_data_point = [[num1, num2]]
prediction = clf.predict(new_data_point)

print("Prediction:", prediction)
