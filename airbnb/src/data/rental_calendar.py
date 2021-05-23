import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class RentalCalendar:
    def __init__(self, data_file:str, city:str):
        self.city = city
        self.df = pd.read_csv(data_file)

    def convert_price(self, df, col='price'):
        """Convert price to numberic values.
        Return:
            new dataframe
        """
        # copy all columns except price
        new_df = df[[c for c in df.columns if c != col]]
        new_df[col] = df[col].apply(lambda x : str(x).replace('$', '').replace(',', '')).astype(float)
        return new_df

    def convert_available(self, df, col='available'):
        """Convert available to numeric type
        """
        new_df = df[[c for c in df.columns if c != col]]
        new_df[col] = df['available'].apply(lambda x: 1 if x == 't' else 0)
        return new_df

    
