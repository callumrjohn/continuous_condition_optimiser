import pandas as pd
from functools import reduce

# Function to merge a list of DataFrames on a specified column with a specified join method
def merge_dfs(df_list, on, how): 
    return reduce(lambda left, right: pd.merge(left, right, on=on, how=how), df_list)