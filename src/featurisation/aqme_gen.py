import ast
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AddHs
from src.utils.config import load_config
from src.utils.data_utils import merge_dfs

# Lists are saved as strings, so we need to convert them back to lists
def str_to_list(val):
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    return val

def import_and_process_aqme(desc_path, smiles_path, id_col='substrate_id', smiles_col='smiles', mol_col='mol'):

    df = pd.read_csv(desc_path)
    smiles_df = pd.read_csv(smiles_path)


    df['code_name'] = df['code_name'].apply(lambda x: x.split('_')[0]) # remove suffix from code_name
    df = df.rename(columns={'code_name': id_col})     # Rename columns for consistency
    df = df.map(str_to_list)
    df = df.sort_values(id_col).reset_index(drop=True)

    # Add molecule objects to the smiles_df and add hydrogens
    smiles_df[mol_col] = smiles_df[smiles_col].apply(Chem.MolFromSmiles)
    smiles_df[mol_col] = smiles_df[mol_col].apply(AddHs)
    smiles_df = smiles_df.sort_values(id_col).reset_index(drop=True)

    if not np.array_equal(df[id_col].values, smiles_df[id_col].values):
        # Ensure both DataFrames have the same id_column values
        raise ValueError("The id_column values of the descriptor DataFrame and the SMILES DataFrame do not match.")

    return df, smiles_df

def separate_atomic_descriptors(df, id_col='substrate_id'):
    # Filter the df to only include columns containing list values
    is_list_col = df.map(lambda x: isinstance(x, list)).any()
    atom_df = df.loc[:, is_list_col]
    mol_df = df.loc[:, ~is_list_col]
    atom_df = pd.concat([df[id_col], atom_df], axis=1, ignore_index=True)
    atom_df.columns = df.columns[:len(atom_df.columns)]

    return atom_df, mol_df

def aromatic_carbons_with_CH(mol):
    return [
        idx for idx in range(mol.GetNumAtoms())
        if (
            mol.GetAtomWithIdx(idx).GetSymbol() == 'C' # Carbon atom
            and mol.GetAtomWithIdx(idx).GetIsAromatic() # Aromatic carbon
            and mol.GetAtomWithIdx(idx).GetTotalNumHs(includeNeighbors=True) > 0 # Aromatic carbon a bonded hydrogen
        )
    ]

# Get the indices of the aromatic carbons with CH for each molecule with a specific min/max descriptor.
def get_minmax_ch_indices(atom_df, smiles_df, descriptor, how ='max', id_col='substrate_id', mol_col='mol'):
    if not np.array_equal(atom_df[id_col].values, smiles_df[id_col].values):
        raise ValueError("The id_col in both DataFrames must match.")

    carbons = [aromatic_carbons_with_CH(mol) for mol in smiles_df[mol_col]]
    carbon_values = [[atom_df[descriptor][i][carbon] for carbon in carbons[i]] for i in range(len(carbons))]  # Get values for aromatic carbons with CH
    if how == 'max':
        return [carbon[np.argmax(carbon_values[i])] for i, carbon in enumerate(carbons)]
    elif how == 'min':
        return [carbon[np.argmin(carbon_values[i])] for i, carbon in enumerate(carbons)]
    elif how == 'both':
        return [np.argmin(carbon_values[i]) for i in range(len(carbons))], [np.argmax(carbon_values[i]) for i in range(len(carbons))]
    else:
        raise ValueError("Invalid value for 'how'. Use 'max', 'min', or 'both'.")


# Generate a new DataFrames with the values corresponding a set of atomic indices.
def get_indexed_carbon_df(atom_df, indexes, id_col='substrate_id'):
    
    column_labels = [col for col in atom_df.columns if col != id_col]  # Exclude the id_col
    
    indexed_values = {f'{col}_indxcarbon' : [atom_df[col][x][indx] for x, indx in enumerate(indexes)] for col in column_labels}

    indexed_df = pd.DataFrame(indexed_values)
    indexed_df = pd.concat([atom_df[[id_col]], indexed_df], axis=1)

    return indexed_df


# Get the min and max values for each descriptor across all atoms in the DataFrame.
def get_abs_minmax_df(atom_df, id_col='substrate_id'):
    """
    Generate a DataFrame with the absolute min and max values for each descriptor across all atoms.
    """
    def _as_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if pd.isna(value):
            return [np.nan]
        return [value]

    min_values = {}
    max_values = {}

    for col in atom_df.columns:
        if col == id_col:
            continue
        min_values[f'{col}_min'] = atom_df[col].apply(lambda x: min(_as_list(x))).tolist()
        max_values[f'{col}_max'] = atom_df[col].apply(lambda x: max(_as_list(x))).tolist()

    abs_minmax_df = pd.DataFrame({**min_values, **max_values})
    abs_minmax_df = pd.concat([atom_df[[id_col]], abs_minmax_df], axis=1)

    return abs_minmax_df


def main():
    config_files = ["configs/base.yaml", "configs/featurisation/aqme.yaml"]
    cfg = load_config(config_files)

    aqme_raw_dir = cfg['output']['aqme_output_dir']
    aqme_csv = cfg['featurisation']['aqme_desc_path']

    desc_path = os.path.join(aqme_raw_dir, aqme_csv)
    #print(f"Descriptor path: {desc_path}")
    smiles_path = cfg['featurisation']['smiles_path']

    id_col = cfg['featurisation']['id_column']
    smiles_col = cfg['featurisation']['smiles_column']
    mol_col = cfg['featurisation']['mol_column']

    descriptor = cfg['featurisation']['chosen_descriptor']
    how = cfg['featurisation']['choice_method']

    output_path = cfg['featurisation']['output_features_path']

    # Import and process AQME descriptors
    atom_df, smiles_df = import_and_process_aqme(desc_path, smiles_path, id_col=id_col, smiles_col=smiles_col, mol_col=mol_col)
    
    # Get atomic descriptors
    atom_df, mol_df = separate_atomic_descriptors(atom_df, id_col=id_col)
    
    # Get the indices of the aromatic carbons with CH for the chosen descriptor
    indexes = get_minmax_ch_indices(atom_df, smiles_df, descriptor, how=how, id_col=id_col, mol_col=mol_col)
    #print(f"Indexes of aromatic carbons with CH for descriptor '{descriptor}': {indexes} of length {len(indexes)}")
    
    # Generate a new DataFrame with the values corresponding to the aromatic carbons with CH
    indexed_df = get_indexed_carbon_df(atom_df, indexes, id_col=id_col)
    
    # Get the absolute min and max values for each descriptor across all atoms
    abs_minmax_df = get_abs_minmax_df(atom_df, id_col=id_col)
    
    # Combine all the dataframes together to create the final set of descriptors
    final_df_to_merge = [mol_df, indexed_df, abs_minmax_df]
    #print(f'Value types in final_df_to_merge: {[type(df) for df in final_df_to_merge]}')
    #print(f"id_col present?: {[id_col in df.columns for df in final_df_to_merge]}")
    final_df = merge_dfs(final_df_to_merge, on=id_col, how='inner')

    #print(f"Final DataFrame shape: {final_df.shape}")
    # Save the final DataFrame to a CSV file
    final_df.to_csv(output_path, index=False)
    print(f"AQME descriptors saved to {output_path}")

if __name__ == "__main__":
    main()