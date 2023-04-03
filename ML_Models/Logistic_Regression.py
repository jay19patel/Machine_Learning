# Import the required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a random binary classification dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the Logistic Regression classifier
logreg = LogisticRegression()

# Fit the Logistic Regression classifier to the training data
logreg.fit(X_train, y_train)

# Predict the classes of the testing data using the trained Logistic Regression classifier
y_pred = logreg.predict(X_test)

# Calculate the accuracy of the Logistic Regression classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
