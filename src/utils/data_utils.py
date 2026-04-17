import pandas as pd
import numpy as np
from functools import reduce
from scipy.special import logit, expit

def merge_dfs(df_list, on, how):
    """
    Merge a list of DataFrames sequentially on a specified column.
    
    Uses the reduce function to perform sequential left-to-right merging of multiple
    DataFrames on a common column, with a specified join type.
    
    Args:
        df_list: List of pandas DataFrames to merge
        on: Column name(s) to merge on (must be present in all DataFrames)
        how: Type of join - 'inner', 'outer', 'left', or 'right'
    
    Returns:
        Merged DataFrame resulting from sequential merging of all input DataFrames
    
    Raises:
        KeyError: If 'on' column(s) not found in any of the DataFrames
    """
    return reduce(lambda left, right: pd.merge(left, right, on=on, how=how), df_list)

def yield_to_unbounded(y):
    """
    Transform yield values from bounded [0, 100] scale to unbounded logit scale.
    
    Converts percentage yield values to unbounded logit-transformed values suitable for
    regression modeling. Values are clipped to avoid extreme values at boundaries.
    
    Args:
        y: Yield values as numpy array or scalar, typically in range [0, 100]
    
    Returns:
        Logit-transformed yield values on unbounded scale
    
    Notes:
        - Values are clipped to [1e-6, 100-1e-6] to avoid infinity in logit transformation
        - Applies logit transformation: log(y/100 / (1 - y/100))
    """
    y = np.clip(y, 1e-6, 100 - 1e-6)  # avoid 0 and 100
    return logit(y / 100)

def unbounded_to_yield(z):
    """
    Transform logit-scaled values back to bounded yield percentage scale [0, 100].
    
    Inverse transformation of yield_to_unbounded, converting unbounded logit-transformed
    values back to percentage yield values in range [0, 100].
    
    Args:
        z: Logit-transformed values on unbounded scale
    
    Returns:
        Yield values back on bounded percentage scale [0, 100]
    
    Notes:
        - Applies inverse logit (sigmoid) transformation: 100 * sigmoid(z)
        - Inverse of yield_to_unbounded transformation
    """
    return 100 * expit(z)