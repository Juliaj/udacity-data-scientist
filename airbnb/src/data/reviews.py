import pandas as pd

import src.data.util as util
import src.data.listings as lst

class Reviews:
    def __init__(self, data_file:str, city:str):
        self.city = city
        self.df = pd.read_csv(data_file)

    def combined_reviews_and_scores(self, listings: lst.Listings):
     
        rev_scores = listings.df

        combined = pd.merge(self.df, rev_scores, how='inner', left_on='listing_id', right_on='id')
        return combined
    
    
