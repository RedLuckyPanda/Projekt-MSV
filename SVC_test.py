from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import arffToArray
import getClasses_MSP

#output = arffToArray.arff_dataToArray("iemo-alle.arff")
output = getClasses_MSP.get_MSP_dataset()
data = output[0]
target = output[1]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1)

#clf = svm.SVC(gamma='scale', C=250.0)
clf = svm.SVC(gamma='scale', C=250.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print("pred: ", np.unique(y_pred, return_counts=True))
print("test: ", np.unique(y_test, return_counts=True))
print("train: ", np.unique(y_train, return_counts=True))
