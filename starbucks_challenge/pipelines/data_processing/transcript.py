"""Process df data. 

df captures customers' activities and interations with offers. The raw data is json formatted with schema:

event (str) - record description (ie transaction, offer received, offer viewed, etc.)
person (str) - customer id
time (int) - time in hours since start of test. The data begins at time t=0
value - (dict of strings) - either an offer id or transaction amount depending on the record

"""
import os
import sys
from typing import Dict, List, Tuple

from pandas.core.series import Series
import numpy as np
import pandas as pd

import data_processing.util as util


def process_values(df) -> Tuple[List[str], List[float], List[float], int]:
    """Separate fields from `value` column into different colummns
    
    Value column is a dict with four different keys: 'reward', 'offer_id', 'amount', 'offer id'.

    Args:
        df: a pandas dataframe with raw data

    Returns:
        offer_ids: List of offer ids
        rewards: List of rewards from each offer
        amounts: transcation amount in $
        null_value_count: number of None encountered

    """

    offer_ids = []
    rewards = []
    amounts = []

    null_value_count = 0
    for v in df['value']:
        if v is None:
            null_value_count += 1
            offer_id, reward, amount = None
        else:
            if 'offer id' in v:
                offer_id = v['offer id']
            elif 'offer_id' in v:
                offer_id = v['offer_id']
            else:
                offer_id = None
            reward = 0. if 'reward' not in v else float(v['reward'])
            amount = 0. if 'amount' not in v else float(v['amount'])
        offer_ids.append(offer_id)
        rewards.append(reward)
        amounts.append(amount)
    return offer_ids, rewards, amounts, null_value_count


def agg_offer_events(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate offer event data by customer_id and offer_id.

    Args:
        df: a pandas dataframe with columns created.

    Return:
        A pandas dataframe aggregated.

    """
    # drop duplicates first -
    new_df = df.copy()
    new_df = new_df.drop_duplicates(subset=['customer_id', 'offer_id'])

    aggd_df = df.groupby(['customer_id', 'offer_id']).agg(
        offer_received_sum=('offer received', 'sum'),
        offer_viewed_sum=('offer viewed', 'sum'),
        offer_completed_sum=('offer completed', 'sum'),
    ).reset_index()

    return aggd_df


def separate_offers_transactions(
        df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separate transcript data to offers and transactions.
    Records with 'offer received', 'offer viewed' or 'offer completed' are related to offers.
    Others will be put into transcations.

    Args:
        df: an input pandas dataframe.
        
    Return:
        A pandas dataframe with offers and transactions.

    """
    # offers
    offer_event_list = ['offer received', 'offer viewed', 'offer completed']
    offer_index = df[df['event'].isin(offer_event_list)].index
    offers_df = df.loc[offer_index, :]

    # transctions
    transaction_index = df[~df['event'].isin(offer_event_list)].index
    transactions_df = df.loc[transaction_index, :]
    return offers_df, transactions_df


def process(df):
    """Goes through all processing steps and transforms raw data into interrim data.

    Args:
        df: A pandas dataframe with raw data input

    Returns:
        A processed dataframe. 
    """

    new_df = df.copy()

    # process 'value' column
    offer_ids, rewards, amounts, null_value_count = process_values(df)
    new_df['offer_id'] = offer_ids
    new_df['reward'] = rewards
    new_df['amount'] = amounts
    new_df.drop('value', axis=1, inplace=True)

    # drop nulls
    if null_value_count > 0:
        # drop records with offer_id as null
        new_df.dropna(axis=1, how=any, subset=['offer_id'])

    # add one-hot encoded columns for 'event'
    events = pd.get_dummies(new_df['event'])
    new_df = pd.concat([new_df, events], axis=1, sort=False)

    # rename columns for standardization
    new_df.rename(columns={'person': 'customer_id'}, inplace=True)

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
              f'as well as the file path to save the cleaned data \n\nExample: python {os.path.basename(__file__)} '\
              '../data/0_raw/profile.json ../data/1_interim/profile.pkl ')


if __name__ == '__main__':
    main()
