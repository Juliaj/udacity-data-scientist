import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Determine whether a sentence starts with a verb.
    """
    def tokenize(self, text:str) -> list:
        """Tokenize text input.

        Arguments:
        -----
            text: input data

        Return:
        -----
            List of normalized and lemmatized tokens
    
        """
        tokens = word_tokenize(text)

        lemmatizer = WordNetLemmatizer()
        clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

        return clean_tokens

    def starting_verb(self, text) -> bool:
        """Returns True if a sentence starts with a verb, otherwise False.

        Arguments:
        -----
            text: raw input text

        Return:
        -----
            Boolean flag

        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(self.tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        """Pass through method.
        """
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """Loop through input dataset and return the boolean flag.

        Arguments:
        -----
            X: input text data 

        Return:
        -----
            Corresponding feature after transformation.

        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
