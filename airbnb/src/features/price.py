"""
Features for rental price prediction
"""
import numpy as np
import pandas as pd
import farmhash

import src.data.listings as lst
import src.data.util as dutil

def combine_listings(pkl_files: str):
    """load processed data pickle files and combine them.
      Parameters:
        pkl_files: path to pickel files
    Return:
        Pandas Dataframe
    """
    dfs = [pd.read_pickle(f) for f in pkl_files]
    return pd.concat(dfs)

def impute_na(df):
    """Impute missing data. 
    Return:
       new Pandas Dataframe 
    """
    new_df = df.copy()
    
    # impute missing values of beds and bathrooms with mean
    for col in ['beds', 'bathrooms', 'bedrooms', 'review_scores_value']:
        new_df = dutil.fill_mean(new_df, col)
    
    # impute missing values of property_type with mode
    new_df = dutil.fill_mode(new_df, 'property_type', mode_idx=0)
    return new_df

def one_hot_encode(df, col):
    """Create one hot encording for col with categorical values
    Return:
        new Pandas Dataframe 
    """
    return pd.get_dummies(df, columns=[col], drop_first=True)

def hash(df, col, bucket_size):
    """hash the categorical values then bucketize. 
    """
    new_df = df[[c for c in df.columns if c != col]]
    new_df[col] = df[col].apply(lambda x: farmhash.fingerprint64(x) % bucket_size) 
    return new_df

def build_features(df, hash_property_type=False, bucket_size=5):
    """Impute missing data and encode categorical colums
    Return:
        new Pandas Dataframe
    """
    new_df = df.copy()
    new_df = impute_na(new_df)
    new_df = one_hot_encode(new_df, col='room_type')
    new_df = hash(new_df, col='property_type', bucket_size=bucket_size) if hash_property_type else one_hot_encode(new_df, col='property_type')
    return new_df

