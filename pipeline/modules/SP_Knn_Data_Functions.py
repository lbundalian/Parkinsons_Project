# ---------------------------------------------------------------------------------------------------
# Scientific Programming Final Project
# By: Linnaeus Bundalian, Judith Osuna, David Cabezas, Martin Kusasira Morgan, Sofía González. 

# Functions for the creation of the KNN model
# ---------------------------------------------------------------------------------------------------

# Import libraries and load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings


# Get data for a KNN classifier
def get_knn_data(df):
    """
    Prepare data to train and test a KNN classifier.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        X_train (pd.DataFrame): Subset of data (predictor variables) for training.
        X_test (pd.DataFrame): Subset of data (target variable) for testing.
        y_train (pd.DataFrame): Subset of labels (predictor variables) for training.
        y_test (pd.DataFrame): Subset of labels (target variable) for testing.
    """
    # Create variable y with the target variable
    y = df['Status']
    
    # Create variable X with the predicting variables
    X = df.drop(['Subject_ID','Status'], axis=1)

    # Split data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test



# Find the best value of n
def find_best_n(n_values, X_train, y_train):
    """
    Plot the accuracy of models with different n values (neighbors).

    Parameters:
        n_values (list): List with potential values of n (neighbors).
        X_train (pd.DataFrame): Subset of data (predictor variables) for training.
        y_train (pd.DataFrame): Subset of labels (predictor variables) for training.

    Returns:
        --
    """

    # Initialize a list to register the models' accuracies
    acc_list = list([])
    
    # Fit models with varying values of n to assess accuracy
    
    with warnings.catch_warnings(): # Suppress warnings within this block
        warnings.filterwarnings("ignore", category=FutureWarning) 
        
        # Fit the models
        for i in n_values:
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            acc_list.append(knn.score(X_train, y_train))
    
    # Plot each model's n and accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(n_values, acc_list)
    plt.xlabel('No. Neighbors')
    plt.ylabel('Model accuracy')



# Create a KNN classifier
def create_knn(n, X_train, y_train):
    """
    Initializes a KNN classifier and fits it to the data.

    Parameters:
        n (int): Number of neighbors for the model.
        X_train (pd.DataFrame): Subset of data (predictor variables) for training.
        y_train (list): Subset of labels (predictor variables) for training.

    Returns:
        knn (model): The KNN classifier model.
    """

    with warnings.catch_warnings(): # Suppress warnings within this block
        warnings.filterwarnings("ignore", category=FutureWarning) 
        
        # Initialize the model with n = neighbors
        knn = KNeighborsClassifier(n_neighbors=n)
        
        ## Fit the model to the data
        knn.fit(X_train, y_train)

    return knn



# Resampling method
def resample(X, y):
    """
    Resamples the X and y parameters for the model.

    Parameters:
        X (pd.DataFrame): Subset of data (predictor variables).
        y (list): Subset of labels (predictor variables).

    Returns:
        X_resampled (pd.DataFrame): X resampled
        y_resampled (list): y resampled.
    """

    with warnings.catch_warnings(): # Suppress warnings within this block
        warnings.filterwarnings("ignore", category=FutureWarning) 
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print("Original train class distribution:", Counter(y))
        print("Resampled train class distribution:", Counter(y_resampled))


    return X_resampled, y_resampled

