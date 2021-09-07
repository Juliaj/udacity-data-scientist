"""Training module for offer response.

Framework used is sklearn.

"""
import os
import sys

from typing import Any

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import time

from joblib import dump

import data_processing.util as util
import offer_response.features as orfeatures

random_forest_clf = RandomForestClassifier(random_state=42)
grad_boost_clf = GradientBoostingClassifier(random_state=42)
ada_boost_clf = AdaBoostClassifier(random_state=42)
CLASSIFER_LIST = [random_forest_clf, grad_boost_clf, ada_boost_clf]


def train(clf, param_grid, X, y, scoring='f1', cv=5, verbose=0):
    """Trains a classifier to the training data using GridSearchCV and f1_score.
    
    Args:
        clf: a classifier to fit
        param_grid: (dict) tuning parameters used with GridSearchCV
        X: pandas DataFrame for input features
        y: pandas Series for training labels
            
    Returns:
        classifer: trained classifier
        best_score: best f1 score   
        time_taken: elapsed time in seconds.
    
    """

    start = time.time()

    grid = GridSearchCV(estimator=clf,
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=cv,
                        verbose=verbose)
    print(f'Training {clf.__class__.__name__} :')
    grid.fit(X.to_numpy(), y.squeeze().to_numpy())
    end = time.time()
    time_taken = round(end - start, 2)

    print(f'Time taken : {time_taken} secs.')
    print(f'Best f1_score : {round(grid.best_score_,4)}')
    print("*" * 40)

    return grid.best_estimator_, grid.best_score_, time_taken


def calc_feature_importance(model, feature_cols):
    feat_imp = pd.DataFrame(model.feature_importances_,
                            index=feature_cols.tolist(),
                            columns=['feat_imp']).reset_index()

    feat_imp.rename(columns={'index': 'feature'}, inplace=True)
    feat_imp['imp_perc'] = np.round(
        (feat_imp['feat_imp'] / feat_imp['feat_imp'].sum()) * 100, 2)
    feat_imp = feat_imp.sort_values(by=['imp_perc'],
                                    ascending=False).reset_index(drop=True)
    feat_imp.drop(columns=['feat_imp'], inplace=True)
    return feat_imp


def get_confusion_matrix(y_true, y_pred, normalized=False):
    """Get confusion matrix from a binary classfier.
            0    1
        ------------
        0   tn   fp
        ------------
        1   fn   tp

    Args:
        y_true: ground truth for target
        y_pred: predicted values for target
        normalized: whether to output number as raw or in percentage

    Returns:
        confusion matrix

    """

    conf_matrix = confusion_matrix(y_true.squeeze().to_numpy(), y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f'true postives: {tp}, false postives: {fp}')
    print(f'true negatives: {tn}, false negatives: {fn}')
    if normalized:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(
            axis=1)[:, np.newaxis]
        print(f"\nNormalized confusion matrix:\n{conf_matrix}")
    return conf_matrix


def save_model(model: Any, model_filepath: str):
    """Persists model to disk.

    Args:
        model: trained model.
        model_filepath: file path to save model.

    Returns:
        None

    """
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:

        input_filepath, output_filepath = sys.argv[1:]
        print(f'Loading feature data from {input_filepath}.')

        offer_response_df = util.load_pkl(input_filepath)
        print(f'    Input data shape: {offer_response_df.shape}')

        X_train, X_test, y_train, y_test = orfeatures.create_training_data(
            offer_response_df, label_col='responded', test_size=0.3)
        print(
            f'    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}'
        )

        print('Training model...')
        model, _, _ = train(random_forest_clf, {}, X_train, y_train)

        print('Calculate confusion matrix ...')
        y_pred = model.predict(X_test)
        get_confusion_matrix(y_test, y_pred, normalized=True).ravel()

        print(f'Saving data to {output_filepath}')
        save_model(model, output_filepath)

        print('Model saved.')

    else:
        print('Please provide the input file path of feature input data '\
              f'as well as the file path to save the model \n\nExample: python {os.path.basename(__file__)} '\
              '../data/1_interim/offer_response.pkl ../output/models/offer_response.pkl ')


if __name__ == '__main__':
    main()
