import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import export_graphviz
from six import StringIO
from src import mohadahybridcode

# %matplotlib inline

print("Executing the SVM Classification Module.....")
print("Using selected features passed by MOHADA module FOR CLASSIFICATION OF PACKETS.....")
# Load the preprocessed data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Train a Support Vector Machine (SVM) classifier using the selected features
clf = SVC(kernel='rbf', random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("--------------------------------------")
print("********Classification Metrics********")
print("ACCURACY:", accuracy)
# Display performance metrics
print("Accuracy: %.2f%%" % (accuracy * 100))
print("Precision: %.2f" % precision)
print("Recall: %.2f" % recall)
print("Confusion Matrix:\n", conf_matrix)
print("--------------------------------------")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()










