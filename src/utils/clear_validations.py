import os
from src.utils.config import load_config


def clear_val_results(dir):
    
    if not os.path.exists(dir):
        print(f"Log directory '{dir}' does not exist.")
        return

    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
                print(f"Removed log directory: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
        
def clear_logs(path):

    if not os.path.exists(path):
        print(f"Log '{path}' does not exist.")
        return
    
    with open(path, 'w') as file:
        print(f"Cleared contents of log file: {path}")



def main():
    config_files = ["configs/base.yaml"]
    cfg = load_config(config_files)

    log_dir = cfg['featurisation']['log_dir']
    val_results_dir = cfg['featurisation']['val_results_dir']

    clear_logs_bool = input("Do you want to clear logs? (yes/no): ").strip().lower()
    if clear_logs_bool == 'yes':
        clear_logs(cfg['featurisation']['log_path'])
    else:
        print("Skipping log clearing.")
    
    clear_val_results_bool = input("Do you want to clear validation results? (yes/no): ").strip().lower()
    if clear_val_results_bool == 'yes':
        clear_val_results(val_results_dir)
    else:
        print("Skipping validation results clearing.")
    
if __name__ == "__main__":
    main()