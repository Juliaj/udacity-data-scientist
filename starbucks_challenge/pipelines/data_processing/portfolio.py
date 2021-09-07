"""Processes a portifolio for offers. 

Portifolio contains offer ids and meta data about each offer (duration, type, etc.).
"""
import os
import sys

import numpy as np
import pandas as pd

import data_processing.util as util


def process_channels_data(df, col='channels'):
    ''' Get the unique channels 
    
    INPUT:
    df - (pandas dataframe) input data
    
    OUTPUT:
    channels_df - (pandas dataframe) with each channel as a seperate column and total channel count
    
    '''

    #Exract unique channels
    channs = set()
    for chan in df[col]:
        channs.update(set(chan))
    channs = list(channs)

    new_channs = pd.DataFrame()
    for column in channs:
        new_channs[f'{column}_ch'] = df[col].apply(lambda x: 1
                                                   if column in x else 0)
        new_channs['num_channels'] = df[col].apply(lambda x: len(x))

    return new_channs


def process(df):
    """Goes through all processing steps and transforms raw data into interrim data.

    Args:
        df: A pandas dataframe with raw data input

    Returns:
        A processed dataframe. 
    """

    new_df = df.copy()

    # convert duration to hours
    new_df['duration'] = df['duration'] * 24

    # split 'channels' into seperate columns
    new_channs = process_channels_data(new_df)
    new_df = pd.concat([new_df, new_channs], axis=1, sort=False)
    # drop column
    new_df.drop('channels', inplace=True, axis=1)

    # apply one-hot encoding to offer_type column
    offer_types = pd.get_dummies(new_df['offer_type'])
    new_df = pd.concat([new_df, offer_types], axis=1, sort=False)
    # drop column
    new_df.drop('offer_type', inplace=True, axis=1)

    # rename for standardization
    new_df.rename(columns={
        'id': 'offer_id',
        'reward': 'offer_reward'
    },
                  inplace=True)

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
        print(f'Please provide the input file path of profile data '\
              f'as well as the file path to save the cleaned data \n\nExample: python {os.path.basename(__file__)} '\
              '../data/0_raw/data1.json ../data/1_interim/data1.pkl ')


if __name__ == '__main__':
    main()
