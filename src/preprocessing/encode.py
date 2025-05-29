import pandas as pd

def one_hot_encode(df, columns, drop_first=True):
    return pd.get_dummies(df, columns=columns, drop_first=drop_first)