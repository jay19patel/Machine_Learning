from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)

# Fit hierarchical clustering model
hierarchical = AgglomerativeClustering(n_clusters=2)
y_pred = hierarchical.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title('Hierarchical Clustering')
plt.show()
