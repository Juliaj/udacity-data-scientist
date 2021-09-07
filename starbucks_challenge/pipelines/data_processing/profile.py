"""Fetches and processes customer profile data.
"""
import sys
from datetime import datetime

from typing import Any, List

import numpy as np
import pandas as pd

import data_processing.util as util

ANOMALOUS_AGE = 118
INCOME_RANGE = range(30000, 150000, 10000)
INCOME_BUCKET_LABELS = [
    '30K', '40K', '50K', '60K', '70K', '80K', '90K', '100K', '110K', '120K',
    '130K'
]


def encode_gender(df: pd.DataFrame, col: str = 'gender') -> pd.DataFrame:
    """Processes missing gender data and split categrories into seperate columns. 

    Args:
        df: A pandas dataframe with raw data input
        col: column name for gender

    Returns:
        A dataframe with gender as one-hot encoded
    """

    return pd.get_dummies(df['gender'])


def group_age(df: pd.DataFrame, col: str = 'age') -> pd.DataFrame:
    """Convert 'age' column into one of the age groups, e.g. 10s, 20s, ... 100s
    
    Args:
        df: a pandas dataframe for input.
        col (optional): name of the age column
    
    Returns:
        age_category_df: a pandas dataframe with ages bucketized into groups.
    
    """
    new_df = df.copy()
    new_df['age_category'] = pd.cut(new_df['age'],
                                    bins=range(10, 120, 10),
                                    right=False,
                                    labels=[
                                        '10s', '20s', '30s', '40s', '50s',
                                        '60s', '70s', '80s', '90s', '100s'
                                    ])

    # apply one-hot encoding
    age_groups = pd.get_dummies(new_df['age_category'],
                                prefix=col,
                                prefix_sep='_')
    return age_groups


def process_became_member_on(df: pd.DataFrame,
                             col: str = 'became_member_on') -> pd.DataFrame:
    """Process 'became_member_on' column into colmuns with year, month and member ship duration.
    
    Args:
        df: a pandas dataframe for input.
        col (optional): name of the column to process.
    
    Returns:
         A pandas dataframe with processed membership info.
    
    """
    new_df = df[[col]].copy()
    new_df['became_member_year'] = pd.to_datetime(new_df[col],
                                                  format='%Y%m%d').dt.year
    new_df['became_member_month'] = pd.to_datetime(new_df[col],
                                                   format='%Y%m%d').dt.month
    became_member_date = pd.to_datetime(new_df[col], format='%Y%m%d').dt.date
    new_df['membership_length'] = (datetime.today().date() -
                                   became_member_date).dt.days

    # apply one-hot encoding for became_member_year and became_member_month
    member_year_dummies = pd.get_dummies(new_df['became_member_year'],
                                         prefix='became_member_year',
                                         prefix_sep='_')
    member_month_dummies = pd.get_dummies(new_df['became_member_month'],
                                          prefix='became_member_month',
                                          prefix_sep='_')

    membership_df = pd.concat(
        [new_df, member_year_dummies, member_month_dummies],
        axis=1,
        sort=False)

    return membership_df


def bucketize_income(df: pd.DataFrame,
                     income_range: Any = INCOME_RANGE,
                     labels: Any = INCOME_BUCKET_LABELS,
                     col: str = 'income') -> pd.DataFrame:
    """bucketize 'income' column into categories.
    
    Args:
        df: a pandas dataframe for input.
        income_range (optional): a range to bucketize the values. 
        labels (optional): A list labels corresponding to the range input.
        col (optional): name of the column to process.
    
    Returns:
        bucketized_df: a pandas dataframe with processed info.
    
    """
    new_df = df[[col]].copy()
    new_df['income_by_range'] = pd.cut(new_df[col],
                                       bins=income_range,
                                       right=False,
                                       labels=labels)

    # apply one-hot encoding
    income_dummies = pd.get_dummies(new_df['income_by_range'],
                                    prefix='income',
                                    prefix_sep='_')

    return income_dummies


def process(df):
    """Goes through all processing steps and transforms raw data into interrim data.

    Args:
        df: A pandas dataframe with raw data input

    Returns:
        A processed dataframe. 
    """

    new_df = df.copy()

    # rename `id` column to `customer_id` for standarization
    new_df.rename(columns={'id': 'customer_id'}, inplace=True)

    # remove records with anomalous age
    index_vals = new_df[new_df['age'] == ANOMALOUS_AGE].index
    new_df.drop(index_vals, inplace=True)

    # process 'became_member_on'
    membership = process_became_member_on(new_df)

    # gender
    genders = encode_gender(new_df, col='gender')

    # process 'age'
    age_groups = group_age(new_df)

    # process 'income'
    income_buckets = bucketize_income(new_df,
                                      income_range=INCOME_RANGE,
                                      labels=INCOME_BUCKET_LABELS,
                                      col='income')

    new_df = pd.concat(
        [new_df, membership, genders, age_groups, income_buckets],
        axis=1,
        sort=False)

    return new_df


def main():

    if len(sys.argv) == 3:

        input_filepath, output_filepath = sys.argv[1:]
        print(f'Loading raw data from {input_filepath}.')

        df = util.load_json(input_filepath)
        print(f'    Input data shape: {df.shape}')

        print('Processing and cleaning data...')
        df = process(df)

        print(f'    After transform, data shape: {df.shape}')

        print(f'Saving data to {output_filepath}')
        util.save(df, output_filepath)

        print('Data saved.')

    else:
        print('Please provide the input file path of profile data '\
              'as well as the file path to save the cleaned data \n\nExample: python profile.py '\
              '../data/0_raw/profile.json ../data/1_interim/profile.pkl ')


if __name__ == '__main__':
    main()
