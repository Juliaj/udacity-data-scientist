"""Provides utilites for data processing modules. 

"""

import sys
from typing import List

import pandas as pd

def load_json(file_path: str) -> pd.DataFrame:
    """Load json data file to dataframe.
    Args:
        data_file: A file name for json formated data.

    Returns:
        A pandas dataframe.
    """

    return pd.read_json(file_path, orient='records', lines=True)

def load_pkl(file_path: str) -> pd.DataFrame:
    """Load pkl data file to dataframe.
    Args:
        data_file: A file name for pkl formated data.

    Returns:
        A pandas dataframe.
    """

    return pd.read_pickle(file_path)

def save(df: pd.DataFrame, file_path:str):
    """Save data to pickel format

    Args:
        df: input data
        file_path: file path to save data to.
        
    Returns:
        None
    """
    df.to_pickle(file_path)

def generate_output_filepath(input_filepath: str , output_dir : str, file_ext : str) -> str:
    """Generete a output file path based on input path.

    The output path will be {out_dir}/{input_file_base}.{out_ext}

    Args:
        input_filepath: an absolute file paths 
        output_dir: absolute directory path
        file_ext: file extension, e.g. json, pkl etc

    Returns:
        An absolute file path.
    """
    pass
    
