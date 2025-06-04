import os
import pandas as pd

def update_log_csv(log_path, row_dict):
    """
    Appends a dictionary as a row to a CSV file, creating the file with headers if it doesn't exist.
    """
    df = pd.DataFrame([row_dict])
    write_header = not os.path.exists(log_path)
    df.to_csv(log_path, mode='a', header=write_header, index=False)