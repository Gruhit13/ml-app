from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.svm import SVC
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
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)

# get prediction from the model
y_pred = clf.predict(X_test)
print("Y_pred shape: ", y_pred.shape)

precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

with open('report.txt', 'w') as report:
	report.write("="*15 + "| ")
	report.write("Support Vector Classifier Report")
	report.write(" |" + "="*15 + "\n")
	report.write(f'Precision: {precision:.4f}\n')
	report.write(f'Recall: {recall:.4f}')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.title("SVC Confusion Matrix", {"size": 22})
plt.xlabel("Predicted", {"size":20})
plt.ylabel("Actual", {"size": 20})
plt.savefig("confusion_matrix.png")

# Saving the model as joblib to parallelize the process
with open('svc_model.sav', 'wb') as f:
	joblib.dump(clf, f)
