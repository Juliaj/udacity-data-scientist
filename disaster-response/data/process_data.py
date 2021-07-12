"""Data Pipeline for disaster response.
"""
import sys
import pandas as pd
import numpy as np
import re
from pandas.core.series import Series
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath:str) -> pd.DataFrame:
    """Load disaster messages and categories raw data and merge.

    Arguments:
    ---------
        messages_filepath: file path to messages csv file.
        categories_filepath: file path to corresponding categories csv file. 

    Returns:
    -------
        Merged dataset. 

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, how='inner', on='id')
    print(f'messages.shape: {messages.shape}, categories.shape: {categories.shape}')
    return df


def transform_categories(df: pd.DataFrame, col: str='categories') -> pd.DataFrame:
    """Process input data and split categrories into seperate columns. 

    Arguments:
    ------
        df: raw data input
        col: column name for categories data

    Return:
    ------
        Processed dataframe with seperated categories into their corresponding columns.
    """
    categories = df[col].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories
    category_colnames = row.apply(lambda x: re.sub('-.*', '', x)).tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
    
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Transform raw categories data and clean data.

    Arguments:
    -------
        df: raw dataframe
    
    Return:
    -------
        Processed dataframe.

    """
    # split categories into separate category columns
    df = transform_categories(df, col='categories')
    
    # remove duplicate and rows missing categories
    df = df.drop_duplicates()
    category_colnames = [c for c in df.columns if c not in ['id', 'message', 'original', 'genre']]
    
    df = df.dropna(how='any', subset=category_colnames)
    
    # replace category value of 2 to 1 
    df['related'] = df.related.astype(int).replace(2, 1)

    return df



def save_data(df: pd.DataFrame, database_filename: str, table_name: str='messages'):
    """Save data into sqllite database

    Arguments:
    ------
        df: input data
        database_filename: path to sqllite database file
        table: name of the table to store data
   
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    conn = engine.connect()

    # drop the messages table in case it already exists
    conn.execute(f'DROP TABLE IF EXISTS {table_name}')

    df.to_sql(f'{table_name}', conn, index=False)
  

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print(f'    Input data shape: {df.shape}')
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'    After transform, data shape: {df.shape}')
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
