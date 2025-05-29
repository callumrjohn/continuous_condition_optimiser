import pandas as pd

def melt_data_df(df, id_vars, var_name = 'variable', value_name = 'value', drop_nan = True):
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


    