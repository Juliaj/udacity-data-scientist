"""Module to predict Airbnb rental price
"""
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def fit_linear_mod(X:pd.DataFrame, y: pd.DataFrame, test_size: float=.3, rand_state: int=42):
# def fit_linear_mod(df:pd.DataFrame, label_col:str, test_size: float=.3, rand_state: int=42):
    '''
    INPUT:
    df - input features
    label_col - the name of the column to predict 
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - seed value

    OUTPUT:
    test_score, train_score - float - r2 score on the test data and train data
    test_rmse, train_rmse - float - rmse on the test data and train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''

    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    lm_model = LinearRegression(normalize=True) 
    lm_model.fit(X_train, y_train) 

    #Predict 
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

    #r2_score 
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)

    #rmse
    test_rmse = mean_squared_error(y_test, y_test_preds, squared=False)
    train_rmse = mean_squared_error(y_train, y_train_preds, squared=False)
    return test_score, train_score, test_rmse, train_rmse, lm_model, X_train, X_test, y_train, y_test


def find_optimal_lm_mod(X, y, cutoffs, test_size =.30, rand_state=42, plot=True):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, labels 
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default True to plot result

    OUTPUT
    best_cutoff - best cutoff for number of non-zero values in dummy categorical vars
    num_feats - total number of features
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    rmse_scores_test - list of root mean squared errors on the test data
    rmse_scores_train - list of root mean squared errors on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    num_feats = []
    r2_scores_test, r2_scores_train, rmse_scores_test, rmse_scores_train = [], [], [], []
    r2_score_results = dict()

    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        r2_test_score, r2_train_score, test_rmse, train_rmse, lm_model, X_train, X_test, y_train, y_test = fit_linear_mod(reduce_X, y, test_size = test_size, rand_state=rand_state)
        
        r2_scores_test.append(r2_test_score)
        r2_scores_train.append(r2_train_score)
        rmse_scores_test.append(test_rmse)
        rmse_scores_train.append(train_rmse)

        # Track cutoff and associated r2 test score
        r2_score_results[str(cutoff)] = r2_test_score
        
    if plot:
        plot_scores(num_feats, r2_scores_test, r2_scores_train)

    best_cutoff = max(r2_score_results, key=r2_score_results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    # num_feats.append(reduce_X.shape[1])
   
    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=rand_state)

    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)

    return num_feats, best_cutoff, r2_scores_test, r2_scores_train, rmse_scores_test, rmse_scores_train, lm_model, X_train, X_test, y_train, y_test

def plot_scores(num_feats, test_scores, train_scores, ylabel='Rsquared'):
    plt.plot(num_feats, test_scores, label="Test", alpha=.5)
    plt.plot(num_feats, train_scores, label="Train", alpha=.5)
    plt.xlabel('Number of Features')
    plt.ylabel(f'{ylabel}')
    plt.title(f'{ylabel} by Number of Features')
    plt.legend(loc=1)
    plt.show()
