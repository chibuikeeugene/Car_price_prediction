import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CarAge(BaseEstimator, TransformerMixin):
    """elaspe time transformer"""

    def __init__(self, variables: str):

        # if not isinstance(variables, int):
        #     raise TypeError('invalid data type, var should be int')
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # this helps fit the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # we create a copy of our dataframe
        X = X.copy()

        # create an empty list and assign to a variable age
        age = []
        for i in X[self.variables].index:

            age.append(X[self.variables].max() - X[self.variables][i])

        X["age"] = age

        return X
