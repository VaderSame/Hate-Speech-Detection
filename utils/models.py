# models.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_preds, normalize=False):
    cm = confusion_matrix(y_true, y_preds)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    names = ['Hate', 'Offensive', 'Neither']
    confusion_df = pd.DataFrame(cm, index=names, columns=names)

    plt.figure(figsize=(5, 5))
    sns.heatmap(confusion_df, annot=True, annot_kws={"size": 12}, cmap='gist_gray_r', cbar=False, square=True, fmt='.2f')
    plt.ylabel('True categories', fontsize=14)
    plt.xlabel('Predicted categories', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.show()


def logistic_regression(X_train, y_train, X_test, y_test):
    # Create a pipeline with Logistic Regression and feature selection
    pipe = Pipeline([
        ('select', SelectFromModel(LogisticRegression(class_weight='balanced', max_iter=1000, penalty="l2", C=0.01))),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000, penalty='l2'))
    ])

    # Define the parameter grid (add parameters if needed)
    param_grid = {
        'select__threshold': ['mean', 'median'],
        'model__C': [0.001, 0.01]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipe, param_grid,
                               cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train),
                               verbose=2)

    # Fit the model
    model = grid_search.fit(X_train, y_train)

    # Make predictions on the test set
    y_preds = model.predict(X_test)

    # Print the classification report
    print("Logistic Regression with Feature Selection Classification Report:")
    print(classification_report(y_test, y_preds))

    # Plot the confusion matrix (raw counts)
    print("\nLogistic Regression with Feature Selection Confusion Matrix (Counts):")
    plot_confusion_matrix(y_test, y_preds)

    # Plot the normalized confusion matrix
    print("\nLogistic Regression with Feature Selection Normalized Confusion Matrix:")
    plot_confusion_matrix(y_test, y_preds, normalize=True)

def linear_svc(X_train, y_train, X_test, y_test):
    # Create a pipeline with LinearSVC
    pipe = Pipeline([('model', svm.LinearSVC(class_weight='balanced', C=1, loss='squared_hinge', multi_class='ovr', max_iter=1000))])

    # Define the parameter grid (add parameters if needed)
    param_grid = [{}]

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipe, param_grid,
                               cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train),
                               verbose=2)

    # Fit the model
    model = grid_search.fit(X_train, y_train)

    # Make predictions on the test set
    y_preds = model.predict(X_test)

    # Print the classification report
    print("LinearSVC Classification Report:")
    print(classification_report(y_test, y_preds))

    # Plot the confusion matrix (raw counts)
    print("\nLinearSVC Confusion Matrix (Counts):")
    plot_confusion_matrix(y_test, y_preds)

    # Plot the normalized confusion matrix
    print("\nLinearSVC Normalized Confusion Matrix:")
    plot_confusion_matrix(y_test, y_preds, normalize=True)

def random_forest(X_train, y_train, X_test, y_test):
    
    # Create a pipeline with RandomForestClassifier and feature selection
    pipe = Pipeline([
        ('select', SelectFromModel(RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42))),
        ('model', RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42))
    ])

    # Define the parameter grid (add parameters if needed)
    param_grid = [{}]  # You can customize this grid if needed

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipe, param_grid,
                               cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train),
                               verbose=2)

    # Fit the model
    model = grid_search.fit(X_train, y_train)

    # Make predictions on the test set
    y_preds = model.predict(X_test)

    # Print the classification report
    print("RandomForest Classification Report:")
    print(classification_report(y_test, y_preds))

    # Plot the confusion matrix (raw counts)
    print("\nRandomForest Confusion Matrix (Counts):")
    plot_confusion_matrix(y_test, y_preds)

    # Plot the normalized confusion matrix
    print("\nRandomForest Normalized Confusion Matrix:")
    plot_confusion_matrix(y_test, y_preds, normalize=True)
        
def multinomial_nb(X_train, y_train, X_test, y_test):
    # Apply Min-Max scaling to ensure non-negative values
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a pipeline with Multinomial Naive Bayes and feature selection
    pipe = Pipeline([
        # ('select', SelectFromModel(MultinomialNB())),  # Uncomment if feature selection is desired
        ('estimator', MultinomialNB())
    ])

    # Define the parameter grid (add parameters if needed)
    param_grid = [{}]  # No additional parameters for this example

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipe, param_grid,
                               cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_scaled, y_train),
                               verbose=2)

    # Fit the model
    model = grid_search.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_preds = model.predict(X_test_scaled)

    # Print the classification report
    print("MultinomialNB with Feature Selection and Min-Max Scaling Classification Report:")
    print(classification_report(y_test, y_preds))

    # Plot the confusion matrix (raw counts)
    print("\nMultinomialNB with Feature Selection and Min-Max Scaling Confusion Matrix (Counts):")
    plot_confusion_matrix(y_test, y_preds)

    # Plot the normalized confusion matrix
    print("\nMultinomialNB with Feature Selection and Min-Max Scaling Normalized Confusion Matrix:")
    plot_confusion_matrix(y_test, y_preds, normalize=True)