from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
labels = list(iris.target_names)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and fit the k-Means model on the training data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)

# Predict the clusters for the test set
y_pred = kmeans.predict(X_test)

# Calculate the silhouette score
silhouette_avg = silhouette_score(X_test, y_pred)
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Create a confusion matrix to compare the cluster assignments with the true labels
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_mat)

# Plot the confusion matrix
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Clusters", {"size": 20})
plt.ylabel("True Labels", {"size": 20})
plt.title("Confusion Matrix of k-Means Clustering", {"size": 20})
plt.savefig("confusion_matrix_kmeans.png")

# Saving the k-Means model using joblib
with open('kmeans_model.sav', 'wb') as f:
    joblib.dump(kmeans, f)
