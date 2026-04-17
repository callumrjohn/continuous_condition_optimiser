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
    """
    Convert string representation of lists back to Python lists.
    
    Attempts to safely convert a string that represents a Python list (e.g., '[1, 2, 3]')
    back to an actual list object using ast.literal_eval. Other data types are returned
    unchanged. This is necessary because CSV files store lists as strings when read.
    
    Args:
        val : any
            The value to convert. Can be a string, list, or any other type.
    
    Returns:
        any
            If val is a string representation of a list, returns the converted list.
            Otherwise returns val unchanged.
    
    Notes:
        Safely handles conversion errors by returning the original value if
        ast.literal_eval fails. This prevents data loss from malformed list strings.
    """
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    return val

def import_and_process_aqme(desc_path, smiles_path, id_col='substrate_id', smiles_col='smiles', mol_col='mol'):
    """
    Import and process AQME descriptors with corresponding SMILES and molecule objects.
    
    Reads AQME descriptor and SMILES CSV files, applies data preprocessing including
    column name standardization, list string conversion, and sorting. Adds RDKit
    molecule objects (with hydrogens) to the SMILES DataFrame for atomic calculations.
    
    Args:
        desc_path : str
            Path to CSV file containing AQME descriptors
        smiles_path : str
            Path to CSV file containing SMILES strings and molecule identifiers
        id_col : str, optional
            Name of the column containing molecule identifiers (default: 'substrate_id')
        smiles_col : str, optional
            Name of the column containing SMILES strings (default: 'smiles')
        mol_col : str, optional
            Name of the column to store RDKit molecule objects (default: 'mol')
    
    Returns:
        tuple
            - df (pd.DataFrame): Processed descriptor DataFrame with all list columns
              from AQME, sorted by id_col
            - smiles_df (pd.DataFrame): DataFrame with SMILES, molecule identifiers,
              and RDKit molecule objects (with hydrogens), sorted by id_col
    
    Raises:
        ValueError: If id_col values don't match between descriptor and SMILES DataFrames
    
    Notes:
        Code names from AQME output (which contain suffixes like '_0', '_1') are
        cleaned to extract only the base substrate ID. Molecule objects include
        explicit hydrogens for accurate atomic-level descriptor calculations.
    """

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
    """
    Separate atomic (list-based) descriptors from molecular (scalar) descriptors.
    
    Splits a DataFrame into two separate DataFrames by checking which columns contain
    list values (atomic descriptors) and which contain scalar values (molecular
    descriptors). This separation is necessary because atomic descriptors must be
    processed differently than molecular descriptors.
    
    Args:
        df : pd.DataFrame
            DataFrame containing both atomic (list) and molecular (scalar) descriptors
        id_col : str, optional
            Name of the column containing molecule identifiers (default: 'substrate_id')
    
    Returns:
        tuple
            - atom_df (pd.DataFrame): DataFrame containing only atomic descriptors
              (list-valued columns) and id_col
            - mol_df (pd.DataFrame): DataFrame containing only molecular descriptors
              (scalar-valued columns) and id_col
    
    Notes:
        The id_col is included in both returned DataFrames. Any column that contains
        at least one list value in any row is considered an atomic descriptor.
    """
    # Filter the df to only include columns containing list values
    is_list_col = df.map(lambda x: isinstance(x, list)).any()
    atom_df = df.loc[:, is_list_col]
    mol_df = df.loc[:, ~is_list_col]
    atom_df = pd.concat([df[id_col], atom_df], axis=1, ignore_index=True)
    atom_df.columns = df.columns[:len(atom_df.columns)]

    return atom_df, mol_df

def aromatic_carbons_with_CH(mol):
    """
    Identify aromatic carbon atoms with bonded hydrogens in a molecule.
    
    Searches the molecule for carbon atoms that are both aromatic and have at least
    one bonded hydrogen atom. These are often reactive sites of interest in organic
    chemistry for halogenation and other substitution reactions.
    
    Args:
        mol : rdkit.Chem.Mol
            RDKit molecule object (typically with explicit hydrogens)
    
    Returns:
        list
            List of atom indices (integers) for aromatic carbons with hydrogen atoms.
            Returns an empty list if no such atoms are found.
    
    Notes:
        Requires molecules with explicit hydrogens for accurate hydrogen count.
        Uses GetIsAromatic() which relies on kekulization/aromaticity perception
        by RDKit.
    """
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
    """
    Get atomic indices of aromatic carbons with minimum/maximum descriptor values.
    
    For each molecule, identifies the aromatic carbons with bonded hydrogens and
    retrieves the index of the carbon with either the maximum or minimum value of
    a specified descriptor. This is useful for identifying reactive sites based on
    computed atomic properties.
    
    Args:
        atom_df : pd.DataFrame
            DataFrame containing atomic descriptors (list-valued columns) indexed
            by molecule identifier
        smiles_df : pd.DataFrame
            DataFrame containing molecule objects indexed by molecule identifier
        descriptor : str
            Name of the descriptor column to use for finding min/max values
        how : str, optional
            Direction of optimization: 'max' (default), 'min', or 'both'.
            'both' returns tuple of (min_indices, max_indices)
        id_col : str, optional
            Name of the column containing molecule identifiers (default: 'substrate_id')
        mol_col : str, optional
            Name of the column containing molecule objects (default: 'mol')
    
    Returns:
        list or tuple
            If how='max': list of atom indices with maximum descriptor values
            If how='min': list of atom indices with minimum descriptor values
            If how='both': tuple of (min_indices list, max_indices list)
    
    Raises:
        ValueError: If DataFrames have mismatched id_col values or invalid 'how' value
    
    Notes:
        Only considers aromatic carbons that have bonded hydrogens. Multiple molecules
        with only one aromatic CH will still return a single value per molecule.
    """
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
    """
    Extract descriptor values at specific atomic indices for each molecule.
    
    Creates a new DataFrame containing only the descriptor values from specified
    atom indices. Given a list of atom indices (one per molecule), this function
    extracts the corresponding values from each atomic descriptor column.
    
    Args:
        atom_df : pd.DataFrame
            DataFrame with atomic descriptors (list-valued columns) indexed by
            molecule identifier. First column should be id_col.
        indexes : list
            List of atom indices, one for each row in atom_df. Each index specifies
            which atom's values to extract for that molecule.
        id_col : str, optional
            Name of the column containing molecule identifiers (default: 'substrate_id')
    
    Returns:
        pd.DataFrame
            DataFrame with columns named '{descriptor}_indxcarbon' containing the
            values from the specified atom index for each descriptor, plus id_col.
            Shape is (n_molecules, n_descriptors + 1)
    
    Notes:
        The resulting column names append '_indxcarbon' suffix to descriptor names
        to indicate they represent indexed carbon atom values.
    """
    
    column_labels = [col for col in atom_df.columns if col != id_col]  # Exclude the id_col
    
    indexed_values = {f'{col}_indxcarbon' : [atom_df[col][x][indx] for x, indx in enumerate(indexes)] for col in column_labels}

    indexed_df = pd.DataFrame(indexed_values)
    indexed_df = pd.concat([atom_df[[id_col]], indexed_df], axis=1)

    return indexed_df


# Get the min and max values for each descriptor across all atoms in the DataFrame.
def get_abs_minmax_df(atom_df, id_col='substrate_id'):
    """
    Generate DataFrame with absolute min and max values for each descriptor across all atoms.
    
    For each atomic descriptor in the input DataFrame, computes the minimum and maximum
    values across all atoms within each molecule. This provides a summary of the range
    of each descriptor's values at the atomic level.
    
    Args:
        atom_df : pd.DataFrame
            DataFrame containing atomic descriptors (list-valued columns). First column
            should be id_col containing molecule identifiers.
        id_col : str, optional
            Name of the column containing molecule identifiers (default: 'substrate_id')
    
    Returns:
        pd.DataFrame
            DataFrame with columns '{descriptor}_min' and '{descriptor}_max' for each
            descriptor, plus id_col. Shape is (n_molecules, 2*n_descriptors + 1)
    
    Notes:
        Handles various container types (lists, tuples, numpy arrays) and NaN values.
        For each descriptor, both minimum and maximum values are computed separately
        across all atomic values for that descriptor within each molecule.
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
    """
    Generate AQME descriptors from input SMILES file and save processed features to CSV.
    
    Main orchestration function that loads configuration, imports AQME descriptors
    and SMILES data, separates atomic and molecular descriptors, identifies key aromatic
    carbons, and extractss descriptor values at those sites. Combines all features
    into a single DataFrame and saves to the output path.
    
    The processing pipeline includes:
    1. Import and preprocess AQME descriptors and SMILES data
    2. Separate atomic (list) and molecular (scalar) descriptors
    3. Find aromatic carbons with hydrogens and their min/max descriptor sites
    4. Extract descriptor values at identified carbon indices
    5. Compute min/max descriptor ranges across all atoms
    6. Merge all feature matrices into a single output DataFrame
    
    Configuration is loaded from 'configs/base.yaml' and 'configs/featurisation/aqme.yaml'
    which specify input/output paths, descriptor and column names, and the choice method
    for selecting aromatic carbons.
    
    Returns:
        None
        Outputs a CSV file with combined AQME features and prints the output path.
    """
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