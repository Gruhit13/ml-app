from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

iris = load_iris()
X, y = iris.data, iris.target

clf = DecisionTreeClassifier()
clf.fit(X, y)

with open('model.pkl', 'wb') as f:
	pickle.dump(clf, f)
