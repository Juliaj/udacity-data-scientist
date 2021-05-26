"""Utility methods.
"""
import numpy as np

def dropna_any(df, cols):
    """Drop rows if any of the columns have value missing
    Parameters:
        cols: columns
    """
    return df.dropna(subset=cols, how='any')

def div(self, x, y):
    """Divide value of col1 by value2 col2
    Parameters:
        values: x, and y
    Return:
        x/y
    """
    return float(x/y) if y !=0 else np.NaN 

def fill_mean(df, col):
    """Fill all missing values with the mean of the column.
    """
    new_df = df.copy()
    fill_mean = lambda col: col.fillna(col.mean()) 
    new_df[col] = new_df[[col]].apply(fill_mean, axis=0)  
    return new_df

def fill_mode(df, col, mode_idx=0):
    """Fill all missing values with the mode of the column.
    """
    new_df = df.copy()
    fill_mode = lambda col: col.fillna(col.mode()[mode_idx])
    new_df[col] = new_df[[col]].apply(fill_mode, axis=0)
    return new_df

def missing_value_count(df, col):
    """Stats of missing values 
    """
    return df[col].isnull().sum() 

def cols_missing_values(df, threshold):
    """Columns with missing values beyond the specified threshold
    """
    most_missing_cols = set(df.columns[df.isnull().mean() > threshold])
    return most_missing_cols

def cols_object_type(df):
    return set(df.select_dtypes(include=['object']))

def cols_no_missing_values(df):
    return set(df.columns[np.sum(df.isnull()) == 0])

def stats(df, df_name):
    num_rows, num_cols = df.shape[0], df.shape[1]
    no_missing_cols = cols_no_missing_values(df)
    most_missing_cols = cols_missing_values(df, 0.75)
    object_type_cols = cols_object_type(df)
    print(f'{df_name} dataset has {num_rows} rows, {num_cols} columns')
    print(f'{len(object_type_cols)} columns has object type')
    print(f'{len(no_missing_cols)} columns without any missing value')
    print(f'{len(most_missing_cols)} columns missing 75% values')

def missing_stats(df, df_name):
    missing = np.sum(df.isnull())
    missing = missing[missing != 0]
    missing = missing.apply(lambda x: f'{round(x*100/df.shape[0],3)} %')
    return missing

def save(df, file_path):
    """Save data to pickel format
    """
    df.to_pickle(file_path)
