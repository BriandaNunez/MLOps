import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Modify this list to include the numerical columns
NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']

# Create custom transformer
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        return self

    def transform(self, X):
        # Put your code here
        X = X.copy()
        for var in self.variables:
            X[var + '_nan'] = pd.isnull(X[var]).astype(int)
        return X


# Read the csv without applying transformations
df = pd.read_csv("module-2/session-6/activity/raw-data.csv")

# Print the first data
print(df.head(10))

mi = MissingIndicator(variables=NUMERICAL_VARS)
# Apply the transformations
df_mi = mi.transform(df)

# Print results after the transformations
print(df_mi.head(20))

