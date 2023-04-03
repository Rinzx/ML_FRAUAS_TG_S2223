# import libraries, modules
# load datasets: training, validation, test

import time
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn import svm, tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve 
import matplotlib.pyplot as plt
from tqdm import tqdm

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

print ('#####...loading models...####')
# create a list of models
models = [
    
    tree.DecisionTreeClassifier(criterion='entropy', splitter='random', min_samples_leaf=5),
    tree.DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_leaf=5),
    tree.DecisionTreeClassifier(criterion='gini', splitter='random', min_samples_leaf=5),
    tree.DecisionTreeClassifier(criterion='gini', splitter='random', min_samples_leaf=10),
]

# train and evaluate each model
for model in models:
    #print (f'model name {model}')
    start_time = time.time()
    print(f'#####...training {type(model).__name__}...####')
    
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy')

    # plot learning curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, np.mean(valid_scores, axis=1), 'o-', color='green', label='Cross-validation score')
    plt.title(f'{type(model).__name__} Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy score')
    plt.legend(loc='best')
    plt.show()

    for _ in tqdm(range(1000000)):
        pass  # a placeholder for training progress

    model.fit(X_train, y_train)
    
    print ('#####...training completed...####')
    
    end_time = time.time()

    total_time = end_time - start_time

    print("Total time taken:", total_time, "seconds")
    
    # Get the current hyperparameters
    params = model.get_params(deep=True)
    
    # Print the hyperparameters
    print ("active hyperparameters:", params)

    # predict on validation data and calculate accuracy
    y_pred_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, y_pred_valid)
    print(f'{type(model).__name__} validation accuracy:', accuracy_valid)

    # predict on test data and calculate accuracy
    y_pred_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f'{type(model).__name__} test accuracy:', accuracy_test)

    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)

    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.set(font_scale=1.4)
    sns.heatmap(cm_norm, annot=True, annot_kws={"size": 16}, cmap="Blues", fmt='.2f')
    
    plt.title(f"{type(model).__name__} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    # Evaluate model performance using accuracy, precision, recall, and F1-score (on test data)
    model_acc = accuracy_score(y_test, y_pred_test)
    model_prec = precision_score(y_test, y_pred_test)
    model_rec = recall_score(y_test, y_pred_test)
    model_f1 = f1_score(y_test, y_pred_test)
    
    print(f'***... {type(model).__name__}...***')
    print("accuracy={:.3f}, precision={:.3f}, recall={:.3f}, F1-score={:.3f}".format(model_acc, model_prec, model_rec, model_f1))

    # Generate ROC curves and calculate AUC scores (Decision Tree)
    model_fpr, model_tpr, model_thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    model_auc = auc(model_fpr, model_tpr)
        
    # Plot ROC curves
    plt.figure()
    plt.plot(model_fpr, model_tpr, label='DT (AUC={:.2f})'.format(model_auc))

    # calculate cross-validation score
    cv_score = cross_val_score(model, X_train, y_train, cv=5)
    print(f'{type(model).__name__} cross-validation score:', cv_score)
    print(f'{type(model).__name__} mean cross-validation score:', cv_score.mean())

print("...  Training finished. !!!")