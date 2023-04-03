# Import the required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the KNN classifier to the training data
knn.fit(X_train, y_train)

# Predict the classes of the testing data using the trained KNN classifier
y_pred = knn.predict(X_test)

# Calculate the accuracy of the KNN classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
