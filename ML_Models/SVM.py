
# Support Vector Machines

# Import the required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the SVM classifier
svm = SVC(kernel='linear')

# Fit the SVM classifier to the training data
svm.fit(X_train, y_train)

# Predict the classes of the testing data using the trained SVM classifier
y_pred = svm.predict(X_test)

# Calculate the accuracy of the SVM classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
