import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(df, columns, drop_first = False, sparse_encoded = False):
    """
    Apply one-hot encoding to specified categorical columns in a DataFrame.
    
    Converts categorical variables into multiple binary columns, one for each
    unique category. Optionally drops the first category to avoid multicollinearity.
    
    Args:
        df: Input DataFrame
        columns: List of column names to one-hot encode
        drop_first: Whether to drop the first category for each feature (default: False).
            Set to True to avoid multicollinearity in linear models.
        sparse_encoded: Whether to return sparse matrix format (default: False)
    
    Returns:
        DataFrame with original non-encoded columns and new one-hot encoded columns.
        Original categorical columns are removed.
    
    Example:
        df = pd.DataFrame({'color': ['red', 'blue', 'red'], 'size': ['S', 'M', 'L']})
        df_encoded = one_hot_encode(df, ['color', 'size'])
    """
    encoder = OneHotEncoder(drop = 'first' if drop_first else None, sparse_output = sparse_encoded)
    encoded = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns), index=df.index)
    df = df.drop(columns=columns)
    df = pd.concat([df, encoded_df], axis=1)
    return df