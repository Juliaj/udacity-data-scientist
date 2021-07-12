"""Generate objects for Plotly plot
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from plotly.graph_objs import Bar
import seaborn as sns


def category_dist(df: pd.DataFrame):
    """Generate plotly json for the message count distribution by categories.

    Arguments:
    -----
    df : input dataframe object.

    """
    category_names = [c for c in df.columns if c not in ['id', 'message', 'original', 'genre']]
    cat_counts = [[c, df[c].astype(int).sum()] for c in category_names]
    # sort by count
    cat_counts = sorted(cat_counts, key=lambda x: x[1], reverse=True)
    categories = [ct[0] for ct in cat_counts]
    counts = [ct[1] for ct in cat_counts]

    graph =   {
            'data': [
                Bar(
                    x=categories,
                    y=counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': {
                       'text': "Category",
                       'standoff': 10
                    },
                    'automargin': 'true',
                },
                'margin':{
                    'b': 200,
                    'autoexpand': 'true'
                }
            }
        }
    return graph

def genre_dist(df):
    """Generate plotly json for the message count distribution by genres.

    Arguments:
    -----
    df : input dataframe object.
    
    """
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    graph =   {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    
    return graph
