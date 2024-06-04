from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

iris = load_iris()
X, y = iris.data, iris.target
clf = LogisticRegression(multi_class="ovr")
clf.fit(X, y)

with open('model.pkl', 'wb') as f:
	pickle.dump(clf, f)
