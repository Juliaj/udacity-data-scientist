"""Disaster message classficaition training module
"""
from typing import Any
import pandas as pd
import numpy as np
import sys
from typing import Tuple

from joblib import dump

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from custom_transformer import StartingVerbExtractor


def load_data(database_filepath: str, table_name:str) -> Tuple[pd.Series, pd.Series, list]:
    """Load data and generate input and labels

    Arguments:
    -----
        database_filepath: sqllite database file
        table_name: table name for input data
    Return:
    -----
        X: input
        y: labels
        category_names: list of categories for prediction

    """
    engine = create_engine(f'sqlite:///{database_filepath}')
 
    df = pd.read_sql_table(table_name, engine)
    category_names = [c for c in df.columns if c not in ['id', 'message', 'original', 'genre']]
    X = df['message']
    y = pd.DataFrame(columns=category_names)
    for col in y.columns:
        y[col] = df[col].astype(int)
    
    return X, y, category_names


def tokenize(text:str) -> list:
    """Tokenize text input.

    Arguments:
    -----
        text: input data

    Return:
    -----
        List of normalized and lemmatized tokens
   
    """
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens


def build_model():
    """Train and tune a mutli-output classifier.

    Arguments
    -----
        X: trainning data
        y: multi-dim labels

    Return:
    -----
        Multioutput Classifier
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('start_verb', StartingVerbExtractor()),
        ])),
            ('clf', MultiOutputClassifier(KNeighborsClassifier(n_neighbors=2), n_jobs=-1)),
        ]
    ) 

    parameters = {
                'features__nlp_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
                'clf__estimator': [KNeighborsClassifier(n_neighbors=2), KNeighborsClassifier(n_neighbors=3)]
             }

    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    return model

def evaluate_model(model, X_test, y_test, category_names, print_report=False) -> Tuple[float, dict]:
    """Evaluate model score and accuracy.

    Arguments:
    -----
        model: trainined model.
        X_test: test data
        y_test: test labels
        category_names: classification categories
        print_report: Print out raw report
    
    Return
    -----
        Model training score, accuracy reports
    """
    score = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    reports = {}
    for i, col in enumerate(category_names):
        y_test_col = y_test.iloc[:, i]
        # y_pred is a numpy array
        y_pred_col = y_pred[:, i]
        if print_report:
            print(f'category = {col}')
            print(classification_report(y_test_col, y_pred_col, zero_division=1))
        report = classification_report(y_test_col, y_pred_col, zero_division=1, output_dict=True)
        reports[col] = report
    return score, reports
  

def save_model(model:Any, model_filepath:str):
    """Persist model.
    Arguments:
    -----
        model: trainned model.
        model_filepath: file path.
    """
    dump(model, model_filepath)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath, table_name='messages')
        print(f'    Input X: {X.shape}, y: {y.shape}, num of labels: {len(category_names)}')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # reset the index for y 
        # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
        y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        score, reports = evaluate_model(model, X_test, y_test, category_names)
        print_scores(score, reports)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
