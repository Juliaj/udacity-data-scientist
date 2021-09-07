from typing import List, Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

COLUMNS_FOR_REMOVAL = [
    'customer_id',
    'offer_id',
    'offer_received_sum',
    'offer_viewed_sum',
    'offer_completed_sum',
    'age',
    'became_member_on',
    'became_member_year',
    'became_member_month',
    'income',
    'gender',
    'purchase_during_offer',
]

FEATURES_TO_SCALE = [
    'difficulty', 'duration', 'offer_reward', 'membership_length'
]


def check_label_balance(y, threshold=10):
    """Check whether the label values are balanced

    Args:
        y: a pandas Series for labels

    Return:
        values percentage: a pandas Series  
    """
    value_counts_perc = round(
        (y.squeeze().value_counts() / y.squeeze().count()) * 100, 2)
    print(f'value stats:\n{value_counts_perc}\n')
    value_std = np.std(value_counts_perc)
    print(f'Values std: {value_std}, threshod: {threshold}.')
    if np.std(value_counts_perc) > threshold:
        print(
            f'WARN: std of values in dataset exceeded threshold, data may not be balanced.'
        )
    else:
        print(f'Values for {y.name} is balanced.')


def clean(offer_response_df, cols_removal=COLUMNS_FOR_REMOVAL):
    """Clean up data to form final feature set.

    Args:
        offer_response_df: a pandas dataframe.
        cols_removal: a list of columns to drop
    Returns:
        A pandas dataframe for features.

    """
    # Drop features from combined_data which are not required for training the model
    new_df = offer_response_df.copy()
    new_df.drop(cols_removal, axis=1, inplace=True)

    # drop nulls
    new_df.dropna(how='any', inplace=True)

    return new_df


def create_training_data(offer_response_df: pd.DataFrame,
                         label_col: str,
                         test_size: float = 0.3,
                         cols_removal: list = COLUMNS_FOR_REMOVAL,
                         features_to_scale: list = FEATURES_TO_SCALE):
    """Create training dataset from persisted offer reseponse dataset

    Args:
        offer_response_df: a pandas dataframe with all data
        label_col: name of column for prediction target 
        test_size: ratio to split data into train and test set
        cols_removal: a list of columns to drop
        features_to_scale: list of features for scaling
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    
    """

    # clean
    feature_df = clean(offer_response_df, cols_removal=cols_removal)

    # check on nulls
    if np.sum(feature_df.isnull().sum()) > 0:
        raise Exception(
            f'Feature dataset has nulls in {np.sum(feature_df.isnull().sum()) } columns.'
        )

    # split data into features X and target y
    X = feature_df.drop(columns=[label_col], axis=1)
    y = feature_df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=42)

    X_train_scaled = scale(X_train, features_to_scale=features_to_scale)
    X_test_scaled = scale(X_test, features_to_scale=features_to_scale)

    return X_train_scaled, X_test_scaled, y_train, y_test


def scale(df, features_to_scale=FEATURES_TO_SCALE):
    """Apply MinMaxScaler to a list of features 
    
    Args:
        - df: a pandas dataframe for input
        - features_to_scale: list of features for scaling
            
    Returns:
        - scaled_df: a pandas dataframe with features scaled

    """

    # Prepare dataframe with features to scale
    df_scaled = df[features_to_scale]

    # Apply feature scaling to df
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled),
                             columns=df_scaled.columns,
                             index=df_scaled.index)

    # Drop orignal features from df and add scaled features
    df = df.drop(columns=features_to_scale, axis=1)
    df_scaled = pd.concat([df, df_scaled], axis=1)

    return df_scaled
