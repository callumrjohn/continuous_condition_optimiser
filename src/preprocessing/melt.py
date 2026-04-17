import pandas as pd

def melt_data_df(df, id_vars, var_name = 'variable', value_name = 'value', drop_nan = True):
    """
    Reshape DataFrame from wide to long format (unpivot operation).
    
    Converts a DataFrame from wide format (multiple columns representing variables)
    to long format (one row per variable value). Optionally removes NaN values
    and converts variable and value columns to numeric types.
    
    Args:
        df: Input DataFrame in wide format
        id_vars: List of column names to use as identifier variables (not melted)
        var_name: Name for the new variable column (default: 'variable')
        value_name: Name for the new value column (default: 'value')
        drop_nan: Whether to remove rows with NaN values in value_name column (default: True)
    
    Returns:
        DataFrame in long format with columns: id_vars, var_name, and value_name
    
    Notes:
        - var_name column is converted to float if possible, otherwise left as-is
        - value_name column is always converted to float
    """
    # Reshape the DataFrame to long format
    long_df = df.melt(
        id_vars = id_vars,
        value_vars = df.columns[len(id_vars):],
        var_name = var_name,
        value_name = value_name
    )
    
    if drop_nan:
        long_df.dropna(subset=[value_name], inplace=True)
    
    # If there variable is continuous, convert it to float
    try:
        long_df[var_name] = long_df[var_name].astype(float)  # Convert tfa_equiv to float
    except ValueError:
        pass
        
    # Convert value column to float
    long_df[value_name] = long_df[value_name].astype(float)
    
    return long_df


    