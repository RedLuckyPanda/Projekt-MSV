#
import numpy as np
import matplotlib.pyplot
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score #gewichtung der features
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import arffToArray
import getClasses_MSP

output = arffToArray.arff_dataToArray("iemo-alle.arff")
#output = getClasses_MSP.get_MSP_dataset()
data = output[0]
target = output[1]
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1)

clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=100, max_depth=50)
clf = clf.fit(data, target)
tree.plot_tree(clf.fit(X_train, y_train))
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print("pred: ", np.unique(y_pred, return_counts=True))
print("test: ", np.unique(y_test, return_counts=True))
print("train: ", np.unique(y_train, return_counts=True))
#print (cross_val_score(clf, iemo_data, iemo_target, cv=62)) #gibt gewichtung der features aus
matplotlib.pyplot.show()