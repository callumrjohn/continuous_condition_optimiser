import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(df, columns, drop_first = False, sparse_encoded = False):
  
    encoder = OneHotEncoder(drop = 'first' if drop_first else None, sparse_output = sparse_encoded)
    encoded = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns), index=df.index)
    df = df.drop(columns=columns)
    df = pd.concat([df, encoded_df], axis=1)
    return df