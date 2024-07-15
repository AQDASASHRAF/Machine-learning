#PREDICTION WITH GRADIENT BOOSTING
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'E:\\data science work\\data science previous laptop work\\table\\parkinson disease dataset.csv'
data = pd.read_csv(file_path)

# Drop the 'name' column as it is not needed for modeling
data = data.drop(columns=['name'])

# Separate features and target variable
X = data.drop(columns=['status'])
y = data['status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = gb_clf.predict(X_test_scaled)
y_pred_proba = gb_clf.predict_proba(X_test_scaled)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Output results
print("Accuracy with gradient boosting:", accuracy)
print("Confusion Matrix using gradient boosting:\n", conf_matrix)
print("ROC AUC:", roc_auc)

# PREDICTION WITH ADABOOST


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset
data_path ='E:\\data science work\\data science previous laptop work\\table\\parkinson disease dataset.csv'
data = pd.read_csv(data_path)

# Drop the 'name' column as it is not needed for modeling
data = data.drop(columns=['name'])

# Separate features and target variable
X = data.drop(columns=['status'])
y = data['status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the AdaBoost Classifier with SAMME algorithm
adaboost_clf = AdaBoostClassifier(algorithm='SAMME', random_state=42)
adaboost_clf.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = adaboost_clf.predict(X_test_scaled)
y_pred_proba = adaboost_clf.predict_proba(X_test_scaled)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Output results
print("Accuracy with Adaboosting:", accuracy)
print("Confusion Matrix using Adaboosting:\n", conf_matrix)
print("ROC AUC:", roc_auc)

# PREDICTION WITH CATBOOST


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'E:\\data science work\\data science previous laptop work\\table\\parkinson disease dataset.csv'
data = pd.read_csv(data_path)

# Drop the 'name' column as it is not needed for modeling
data = data.drop(columns=['name'])

# Separate features and target variable
X = data.drop(columns=['status'])
y = data['status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the CatBoost Classifier
catboost_clf = CatBoostClassifier(random_state=42, verbose=0)
catboost_clf.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = catboost_clf.predict(X_test_scaled)
y_pred_proba = catboost_clf.predict_proba(X_test_scaled)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Output results
print("Accuracy with Catboosting:", accuracy)
print("Confusion Matrix using Catboosting:\n", conf_matrix)
print("ROC AUC:", roc_auc)
