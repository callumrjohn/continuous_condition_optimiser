import pandas as pd
import numpy as np

def find_target_range(df, column):
  
    if column not in df.columns:
        raise ValueError(f"Variable '{column}' not found in DataFrame columns.")

    min_val = df[column].min()
    max_val = df[column].max()

    return min_val, max_val