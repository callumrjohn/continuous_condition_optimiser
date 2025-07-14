import pandas as pd
import numpy as np
from functools import reduce
from scipy.special import logit, expit

# Function to merge a list of DataFrames on a specified column with a specified join method
def merge_dfs(df_list, on, how): 
    return reduce(lambda left, right: pd.merge(left, right, on=on, how=how), df_list)

# Convert yield to unbounded logit scale and back
def yield_to_unbounded(y):
    y = np.clip(y, 1e-6, 100 - 1e-6)  # avoid 0 and 100
    return logit(y / 100)

def unbounded_to_yield(z):
    return 100 * expit(z)