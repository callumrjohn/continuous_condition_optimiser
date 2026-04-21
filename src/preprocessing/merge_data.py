import os
import pandas as pd
from glob import glob
from functools import reduce
from src.utils.config import load_config
from src.utils.data_utils import merge_dfs
from src.guis.fingerprint_selector import select_fingerprints_tkinter


def gen_merge_dfs(data, fingerprints, id_col, how='inner', duplicate_selection = 'first', desc_labels=None):
    """
    Merge experimental data with multiple descriptor/fingerprint DataFrames.
    
    Sequentially merges multiple descriptor/fingerprint DataFrames with the main
    experimental data on a common ID column. Handles duplicate columns that appear
    in multiple descriptor sets by keeping first, last, or averaging them.
    
    Args:
        data: Main experimental data DataFrame
        fingerprints: List of descriptor/fingerprint DataFrames to merge
        id_col: Column name to merge on (present in all DataFrames)
        how: Type of merge - 'inner', 'outer', 'left', 'right' (default: 'inner')
        duplicate_selection: How to handle duplicate column names:
            - 'first': keep first occurrence
            - 'last': keep last occurrence
            - 'mean': average values (not yet implemented)
        desc_labels: Labels for each fingerprint set to append to column names.
            If None, uses range(len(fingerprints))
    
    Returns:
        Merged DataFrame with data and selected descriptor columns
    
    Raises:
        ValueError: If invalid 'how', 'duplicate_selection', or no fingerprints provided
    """
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

    if len(renamed_fingerprints) == 0:
        raise ValueError("No fingerprints provided for merging.")

    elif len(renamed_fingerprints) == 1:
        merged_df = pd.merge(data, renamed_fingerprints[0], on=id_col, how=how)
        return merged_df

    merged_desc = merge_dfs(renamed_fingerprints, id_col, how=how)

    # Handle duplicate descriptors based on the specified method when combining fingerprints
    # Find base column names (without suffix)
    base_names = {}
    for col in merged_desc.columns:
        if col == id_col:
            continue
        else:
            base = col.rsplit('_', 1)[-2]   # Get the base name before the last underscore added through labelling
            base_names.setdefault(base, []).append(col)
    
    cols_to_keep = [id_col]

    for base, cols in base_names.items():
        if len(cols) == 1 or duplicate_selection == 'first':
            cols_to_keep.append(cols[0])

        elif duplicate_selection == 'last':
            cols_to_keep.append(cols[-1])
        
        elif duplicate_selection == 'mean':
            return print("'mean' selection is not implemented yet. Please choose 'first' or 'last'.")

    #print(cols_to_keep)
    merged_desc = merged_desc[cols_to_keep]
    merged_df = pd.merge(data, merged_desc, on=id_col, how=how)

    return merged_df


def remove_nan_columns(df, how='all'):
    """
    Remove columns with all NaN values from the DataFrame.
    """
    return df.dropna(axis=1, how=how)


def remove_no_variance_columns(df):
    """
    Remove columns with no variance (constant columns) from the DataFrame.
    """
    return df.loc[:, df.nunique(dropna=False) > 1]


def main():
    """
    Main entry point for merging experimental data with descriptor/fingerprint sets.
    
    Launches an interactive GUI for users to select which descriptor/fingerprint sets
    to merge together. Optionally removes NaN and no-variance columns. Saves the
    merged dataset with a descriptive filename.
    
    Returns:
        None (saves merged DataFrame to CSV file)
    """
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

    # Column deletion parameters
    remove_nan_columns_flag = cfg['preprocessing']['remove_nan_columns']
    remove_no_variance_columns_flag = cfg['preprocessing']['remove_no_variance_columns']
    
    
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

    merged_df = gen_merge_dfs(data, selected_fingerprints, id_col=id_col, how=how, duplicate_selection=duplicate_selection, desc_labels=selected_names)

    if remove_nan_columns_flag:
        how = cfg['preprocessing']['nan_removal_method']
        if how not in ['all', 'any']:
            raise ValueError("Invalid 'how' parameter for NaN removal. Choose from 'all' or 'any'.")
        merged_df = remove_nan_columns(merged_df, how=how)
        print(f"Removed columns with {how} NaN values.")

    if remove_no_variance_columns_flag:
        merged_df = remove_no_variance_columns(merged_df)
        print(f"Removed columns with no variance.")


    file_name = f"data_{'_'.join(selected_names)}.csv"
    print(file_name)
    output_path = os.path.join(output_dir, file_name)
    print(output_path)
    merged_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()