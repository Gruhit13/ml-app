from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

iris = load_iris()
X, y = iris.data, iris.target
labels = list(iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size = 0.1
)

# Train the model
clf = XGBClassifier(objective='multi:softprob')
clf.fit(X_train, y_train)

# get prediction from the model
y_pred = clf.predict(X_test)
print("Y_pred shape: ", y_pred.shape)

clf_report = classification_report(y_test, y_pred, target_names=labels)
print(clf_report)

# Write the classification report to a file
with open('clf_report.txt', 'w') as clf_rprt_file:
	clf_rprt_file.write(clf_report)

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted", {"size":20})
plt.ylabel("Actual", {"size": 20})
plt.savefig("confusion_matrix.png")

# Saving the model as joblib to parallelize the process
with open('model.sav', 'wb') as f:
	joblib.dump(clf, f)
