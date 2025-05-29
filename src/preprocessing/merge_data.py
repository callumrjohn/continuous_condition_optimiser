import os
import pandas as pd
from glob import glob
from functools import reduce
from src.utils.config import load_config
from src.guis.fingerprint_selector import select_fingerprints_tkinter


def merge_dfs(data, fingerprints, id_col, how='inner', duplicate_selection = 'first', desc_labels=None):

    if how not in ['inner', 'outer', 'left', 'right']:
        raise ValueError("Invalid 'how' parameter. Choose from 'inner', 'outer', 'left', or 'right'.")
    
    if duplicate_selection not in ['first', 'last', 'mean']:
        raise ValueError("Invalid 'duplicate_selection' parameter. Choose from 'first', 'last', 'mean'.")
    
    if desc_labels is None:
        desc_labels = range(len(fingerprints))

    renamed_fingerprints = []
    for i, df in enumerate(fingerprints):
        # Rename columns to avoid conflicts
        renamed_df = df.rename(columns=lambda x: f"{x}_{desc_labels[i]}" if x != id_col else id_col)
        renamed_fingerprints.append(renamed_df)
    to_merge = [data] + renamed_fingerprints

    merged = reduce(lambda left, right: pd.merge(left, right, on=id_col, how=how), to_merge)

    # Handle duplicate sections based on the specified method
    
    # Find base column names (without suffix)
    base_names = {}
    for col in merged.columns:
        if col == id_col:
            continue
        if '_' in col:
            base = col.rsplit('_', 1)[0]
            base_names.setdefault(base, []).append(col)
        else:
            base_names.setdefault(col, []).append(col)

    cols_to_keep = [id_col]
    new_cols = {}

    for base, cols in base_names.items():
        if len(cols) == 1:
            cols_to_keep.append(cols[0])
        else:
            if duplicate_selection == 'first':
                selected = min(cols, key=lambda c: merged.columns.get_loc(c))
                cols_to_keep.append(selected)
            elif duplicate_selection == 'last':
                selected = max(cols, key=lambda c: merged.columns.get_loc(c))
                cols_to_keep.append(selected)
            elif duplicate_selection == 'mean':
                return print("'mean' selection is not implemented yet. Please choose 'first' or 'last'.")
                # Try to convert to numeric, ignore errors
                try:
                    vals = merged[cols].apply(pd.to_numeric, errors='coerce')
                    new_col = vals.mean(axis=1)
                    new_cols[base] = new_col
                    cols_to_keep.append(base)
                except Exception:
                    # If conversion fails, just keep the first
                    selected = min(cols, key=lambda c: merged.columns.get_loc(c))
                    cols_to_keep.append(selected)

        merged_df = merged.copy()
        for base, col_data in new_cols.items():
            merged_df[base] = col_data

        merged_df = merged_df[cols_to_keep]



    return merged_df


def main():
    
    # Load configs
    config_files = ["configs/base.yaml", "configs/preprocessing/merge_data.yaml"]
    cfg = load_config(config_files)

    # Input and output paths
    data_input_path = cfg['data']['yield_path']
    fingerprints_input_dir = cfg['data']['features_dir']
    output_dir = cfg['data']['model_input_dir']

    # Merge parameters
    id_col = cfg['featurisation']['id_column']
    how = cfg['preprocessing']['join_type']
    duplicate_selection = cfg['preprocessing']['duplicate_selection']
    #desc_labels = cfg['preprocessing']['desc_labels']
    
    
    fingerprints_input_paths = glob(fingerprints_input_dir + "/*.csv")
    fingerprint_names = [os.path.splitext(os.path.basename(path))[0].split('_')[0] for path in fingerprints_input_paths]
    print(f"Found fingerprints/descriptors: {fingerprint_names}")
    
    #launch fingerprint selection GUI
    print("Launching fingerprint selector...")
    selected_indices = select_fingerprints_tkinter(fingerprint_names)
    #print(f"Selected fingerprints indices: {selected_indices}")
    selected_fingerprints = [pd.read_csv(fingerprints_input_paths[i]) for i in selected_indices]
    selected_names = [fingerprint_names[i] for i in selected_indices]
    
    print(f"Selected fingerprints:")
    for i, name in enumerate(selected_names):
        print(f"{i+1}: {name}")

    

    # Read data
    data = pd.read_csv(data_input_path)
    if id_col not in data.columns:
        raise ValueError(f"ID column '{id_col}' not found in data file: {data_input_path}")
    
    merged_df = merge_dfs(data, selected_fingerprints, id_col=id_col, how=how, duplicate_selection=duplicate_selection, desc_labels=selected_names)
    
    file_name = f"data_{'_'.join(selected_names)}.csv"
    print(file_name)
    output_path = os.path.join(output_dir, file_name)
    print(output_path)
    merged_df.to_csv(output_path, index=False)


if __name__ in "__main__":
    main()