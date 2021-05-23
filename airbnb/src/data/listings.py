import numpy as np
import pandas as pd

from icecream import ic

import src.data.util as util

class Listings:
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
        new_df[col] = df[col].apply(lambda x : x.replace('$', '').replace(',', '')).astype(float)
        return new_df

    def process_cleaning_fee(self, df, col='cleaning_fee'):
        """Process clean fee and convert data into float.
        Return:
            new dataframe
        """
        # copy all columns except the cleaning fee
        new_df = df[[c for c in df.columns if c != col]]
        
        # fill na with zero
        new_df[col] = df[col].fillna('0')
        new_df[col] = new_df[col].apply(lambda x : x.replace('$', '').replace(',', '')).astype(float)
        return new_df
        
    def scale_price(self, df, by_col, new_col, price_col='price_numeric'):
        """Scale price to single unit of scale by column. 
        Parameters:
            by_col: column name to scale price, e.g. accomodates
            new_col: prefix for new column name 
            price_col: column name for price
        Return:
            new dataframe
        """
        self.df[new_col] = self.df[[price_col, by_col]].apply(lambda x: util.div(*x), axis=1)

    def process_zipcode(self, df, col='zipcode'):
        """Process zipcode column and convert to int.
        Return:
            new dataframe
        """
        new_df = df.copy()
        new_df = new_df.dropna(subset=[col], how='any', axis=0)
        # handle rows with multiple zip codes
        new_df.loc[:, col] = new_df[col].apply(lambda x: self.parse_zipcode(x))
        return new_df
    
    def parse_zipcode(self, x):
        """Parse zipcode string and return zipcode
        """
        if '\n' in x:
            zipcode = x.split('\n')[1]
        else:
            zipcode = x.split(" ")[0].split("-")[0]
        
        try:
            converted_zipcode = int(zipcode)
        except Exception as exc:
            converted_zipcode = 0
        return converted_zipcode

    def add_amenities_count(self, df, col='amenities', drop_amenities=True):
        """Add a colum with the count number of amenties provided.
        Return:
            new dataframe
        """
        new_df = df.copy()
        new_df[f'{col}_count'] = df[col].apply(lambda x: len(x.split(",")))
        if drop_amenities:
            new_df = new_df.drop(columns=['amenities'], axis=1)
        return new_df

    def plot_price_with(self, title, df, x_col, y_col='price'):
        """Scatter plot to identify correlation of a column to price.
        NaN values are excluded
        """
        rental_price = pd.DataFrame()
        rental_price = df[[x_col, y_col]].dropna()
        ax = rental_price.plot.scatter(x_col, y_col)
        ax.set(xlabel=x_col, ylabel='price in dollars', title=title)

    def generate_price_pred_dataset(self):
        """Generate dataset for rental price prediction
        Return:
            new dataframe
        """
        cols = ['zipcode', 'property_type',	'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'amenities', 'price', 'review_scores_value']
        # convert a few columns to numeric types
        new_df = self.process_zipcode(self.df[cols])
        new_df = self.add_amenities_count(new_df)
        new_df = self.convert_price(new_df)
        return new_df

    def generate_review_score_dataset(self):
        """Generate dataset for review score related info
        """
        rev_cols = ['id', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_value']
        return self.df[rev_cols]

    
        

