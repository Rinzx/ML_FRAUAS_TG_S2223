# Import modules

import time
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV


# load the data
# split the data into features and target variables; X and y respectively for each subset
print ('#####...loading training data...####')
X_train = pd.read_excel('C:\\Users\\QuickPass\\Documents\\ML\\Sorted\\X_train.xlsx') 
y_train = pd.read_excel('C:\\Users\\QuickPass\\Documents\\ML\\Sorted\\y_train.xlsx').values.ravel()

print ('#####...loading validation data...####')
X_valid = pd.read_excel('C:\\Users\\QuickPass\\Documents\\ML\\Sorted\\X_val.xlsx')
y_valid = pd.read_excel('C:\\Users\\QuickPass\\Documents\\ML\\Sorted\\y_val.xlsx').values.ravel()

print ('#####...loading training data...####')
X_test = pd.read_excel('C:\\Users\\QuickPass\\Documents\\ML\\Sorted\\X_test.xlsx')
y_test = pd.read_excel('C:\\Users\\QuickPass\\Documents\\ML\\Sorted\\y_test.xlsx').values.ravel()

print ("#####.... data subsets are ready for feature extraction...#####")

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

print("#####... data standardization finished! ...#####")

# Get the parameters used for the standardization
print (scaler.get_params(deep=True))

# Create a Decision Tree classifier object
dtc = tree.DecisionTreeClassifier()

# Define the parameters for the Decision Tree model
parameters = {'max_depth': [2, 4, 6, 8, 10, None],
              'criterion': ['gini','entropy'],
              'min_samples_split': [2, 4, 6, 8, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}


# Create a GridSearchCV object
clf = GridSearchCV(dtc, parameters)

# Fit the GridSearchCV object on the training set
clf.fit(X_train, y_train)

# Print the best estimator and its score on the validation set
print('Best estimator:', clf.best_estimator_)
print('Best parameters:', clf.best_params_)
print('Accuracy on validation set:', clf.score(X_valid, y_valid))

# Calculate the score on the testing set
test_score = clf.score(X_test, y_test)
print('Accuracy on testing set:', test_score)

# Extract the results of the grid search
cv_results = clf.cv_results_
print (cv_results)

cv_results = clf.cv_results_
mean_test_scores = cv_results['mean_test_score']
params = cv_results['params']
max_depth = [params[i]['max_depth'] for i in range(len(params))]
min_samples_split = [params[i]['min_samples_split'] for i in range(len(params))]
min_samples_leaf = [params[i]['min_samples_leaf'] for i in range(len(params))]

print (mean_test_scores)