import os
import pandas as pd
from src.utils.config import load_config


def archive_val_log(val_log_path, val_log_archive_dir=None):

    if not os.path.exists(val_log_path):
        print(f"Log directory '{val_log_path}' does not exist.")
        return

    if val_log_archive_dir is None:
        print("No archive directory specified.")
        return
    
    os.makedirs(val_log_archive_dir, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    new_val_log_path = os.path.join(val_log_archive_dir, f"validation_log_pre_{timestamp}.csv")
    os.rename(val_log_path, new_val_log_path)

    try:
        os.remove(val_log_path)
    
    except Exception as e:
        print(f"Error removing {val_log_path}: {e}")
        

def main():
    config_files = ["configs/base.yaml"]
    cfg = load_config(config_files)
    val_log_path = cfg['output']['val_log_path']
    val_log_archive_dir= cfg['output']['val_log_archive_dir']

    archive_val_log(val_log_path, val_log_archive_dir)
    
if __name__ == "__main__":
    main()