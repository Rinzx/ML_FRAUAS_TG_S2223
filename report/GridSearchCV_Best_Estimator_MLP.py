# Import modules

import time
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# load the data
# split the data into features and target variables
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

# Create the model object and define parametes for the GridSearchCV
# Create an MLP classifier object
mlp = MLPClassifier(solver='adam', max_iter=1000)

# Define the parameters for the MLP model
parameters = {'hidden_layer_sizes': [(10,), (50,), (100,), (10, 10), (50, 50), (100, 100)],
              'activation': ['relu', 'logistic'],
              'alpha': [0.0001, 0.001, 0.01, 0.1]}

# Create a GridSearchCV object
clf = GridSearchCV(mlp, parameters)

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

mean_test_scores = cv_results['mean_test_score']
params = cv_results['params']
hidden_layer_sizes = [params[i]['hidden_layer_sizes'] for i in range(len(params))]
activation = [params[i]['activation'] for i in range(len(params))]
alpha = [params[i]['alpha'] for i in range(len(params))]

print (mean_test_scores)