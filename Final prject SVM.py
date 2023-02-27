import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv(r"C:\Users\anura\Desktop\diabetes.csv")

# Split the dataset into features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Select the best k features using F-score
selector = SelectKBest(f_classif, k=5)
X = selector.fit_transform(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform grid search to find the best hyperparameters for the SVM classifier
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Train the SVM classifier with the best hyperparameters
svm = grid_search.best_estimator_
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)
    
# Calculate the accuracy, specificity, and sensitivity of the SVM classifier
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print('Accuracy:', accuracy)
print('Specificity:', specificity)
print('Sensitivity:', sensitivity)


