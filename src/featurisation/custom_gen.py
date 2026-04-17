import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from src.utils.config import load_config

'''SULFUR NUCLEOPHILES-----------------------------------------------'''

def has_thiol(mol):
    """Check if a molecule contains a thiol group (-SH)."""
    thiol_smarts = Chem.MolFromSmarts('[SX2H]([C;!$(C=O)])')  # Excludes thiols with carbonyls
    return mol.HasSubstructMatch(thiol_smarts)

def has_thioether(mol):
    """Check if a molecule contains a thioether group (R-S-R')."""
    thioether_smarts = Chem.MolFromSmarts('[SX2]([C;!$(C=O)])[C;!$(C=O)]') # does NOT include thioethers with carbonyls or aromatic sulfures
    return mol.HasSubstructMatch(thioether_smarts)

def has_thioester(mol):
    """Check if a molecule contains a thioester group (R-S-C(=O)-R')."""
    thioester_smarts = Chem.MolFromSmarts('[C](=O)[S][#6]')
    return mol.HasSubstructMatch(thioester_smarts)

def has_thiocarboxylic_acid(mol):
    """Check if a molecule contains a thiocarboxylic acid group (R-S-C(=S)-OH)."""
    thiocarboxylic_acid_smarts = Chem.MolFromSmarts('[C](=O)[SX2H]')
    return mol.HasSubstructMatch(thiocarboxylic_acid_smarts)

def has_sulfoxide(mol):
    """Check if a molecule contains a sulfoxide group (R-S(=O)-R')."""
    sulfoxide_smarts = Chem.MolFromSmarts('[S](=O)[#6]')
    return mol.HasSubstructMatch(sulfoxide_smarts)

'''OTHER INCOMPATABLE MOIETIES?--------------------------------------'''

def has_alkene(mol):
    """Check if a molecule contains an alkene group (C=C)."""
    alkene_smarts = Chem.MolFromSmarts('C=C')
    return mol.HasSubstructMatch(alkene_smarts)

def has_alkyne(mol):
    """Check if a molecule contains an alkyne group (C#C)."""
    alkyne_smarts = Chem.MolFromSmarts('C#C')
    return mol.HasSubstructMatch(alkyne_smarts)

def has_tertiary_alcohol(mol):
    """Check if a molecule contains a tertiary alcohol group."""
    tertiary_alcohol_smarts = Chem.MolFromSmarts('[CX4](O)([C,c])([C,c])[C,c]')
    return mol.HasSubstructMatch(tertiary_alcohol_smarts)

def has_glycosamine(mol):
    """Check if a molecule contains a glycosamine group."""
    glycosamine_smarts = Chem.MolFromSmarts('[C;!$(C=O)]([O])[N,n]')
    return mol.HasSubstructMatch(glycosamine_smarts)

def has_glycoside(mol):
    """Check if a molecule contains a glycoside group."""
    glycoside_smarts = Chem.MolFromSmarts('[C;!$(C=O)]([O])[O,o]')
    return mol.HasSubstructMatch(glycoside_smarts)



def get_custom_descriptors(smiles):
    """
    Generate binary features indicating presence of specific chemical functional groups.
    
    Analyzes a molecule for the presence of 10 predefined functional groups related to
    reactivity and chemical compatibility. Returns a dictionary of boolean values
    indicating which functional groups are present.
    
    Args:
        smiles : str
            SMILES string representing the molecule to analyze
    
    Returns:
        dict or None
            Dictionary with 10 boolean keys:
            'has_thiol', 'has_thioether', 'has_thioester', 'has_thiocarboxylic_acid',
            'has_sulfoxide', 'has_alkene', 'has_alkyne', 'has_tertiary_alcohol',
            'has_glycosamine', 'has_glycoside'
            Returns None if SMILES parsing fails.
    
    Notes:
        Invalid SMILES strings print a warning and return None. Functional groups
        are detected using SMARTS pattern matching with RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES string: {smiles}")
        return None
    # Initialize a dictionary to hold custom features

        
    custom_features = {}

    custom_features['has_thiol'] = has_thiol(mol)
    custom_features['has_thioether'] = has_thioether(mol)
    custom_features['has_thioester'] = has_thioester(mol)
    custom_features['has_thiocarboxylic_acid'] = has_thiocarboxylic_acid(mol)
    custom_features['has_sulfoxide'] = has_sulfoxide(mol)
    custom_features['has_alkene'] = has_alkene(mol)
    custom_features['has_alkyne'] = has_alkyne(mol)
    custom_features['has_tertiary_alcohol'] = has_tertiary_alcohol(mol)
    custom_features['has_glycosamine'] = has_glycosamine(mol)
    custom_features['has_glycoside'] = has_glycoside(mol)

    return custom_features



def gen_custom_descriptors(df, smiles_col, id_col):
    """
    Generate binary custom feature DataFrame from SMILES strings for multiple molecules.
    
    Applies get_custom_descriptors() to each SMILES string in the DataFrame,
    collecting binary indicator features for 10 functional groups. Handles invalid
    SMILES gracefully and tracks failed molecules.
    
    Args:
        df : pd.DataFrame
            Input DataFrame containing SMILES strings and molecule identifiers
        smiles_col : str
            Name of the column containing SMILES strings
        id_col : str
            Name of the column containing molecule identifiers
    
    Returns:
        tuple
            - desc_df (pd.DataFrame): DataFrame with binary custom descriptor columns
              plus id_col. Each row represents one molecule. Columns ordered as
              [id_col, 'has_thiol', 'has_thioether', ...] etc.
            - failed (list): List of molecule IDs for which SMILES parsing failed
    
    Notes:
        Empty descriptor dictionaries are added for molecules with invalid SMILES,
        which may result in all-False values for those rows.
    """
    descriptors = []
    failed = []

    for _, row in df.iterrows():
        
        desc = get_custom_descriptors(row[smiles_col])
        
        if desc is None:
            failed.append(row[id_col])
            descriptors.append({})
      
        else:
            descriptors.append(desc)

    # Assemble DataFrame
    desc_df = pd.DataFrame(descriptors)
    columns = desc_df.columns.tolist()
    desc_df[id_col] = df[id_col]

    # Reorder columns
    cols = [id_col] + columns
    return desc_df[cols], failed


def main():
    """
    Generate custom binary descriptors from input SMILES file and save to CSV.
    
    Loads configuration from YAML files, reads SMILES data from input CSV,
    generates custom binary descriptors (encoding presence/absence of specific
    chemical functionalities) using gen_custom_descriptors(), and saves the
    resulting feature matrix to the specified output path.
    
    Configuration is loaded from 'configs/base.yaml' and 'configs/featurisation/custom.yaml'
    which specify input/output paths and SMILES/ID column names.
    
    The custom descriptors check for the presence of functional groups including:
    thiols, thioethers, thioesters, thiocarboxylic acids, sulfoxides, alkenes,
    alkynes, tertiary alcohols, glycosamines, and glycosides.
    
    Returns:
        None
        Outputs a CSV file with custom binary descriptors and prints the output path.
        Also warns about any molecules with invalid SMILES strings.
    """
    config_files = ["configs/base.yaml", "configs/featurisation/custom.yaml"]
    cfg = load_config(config_files)

    input_path = cfg['featurisation']['smiles_path']
    smiles_col = cfg['featurisation']['smiles_column']
    id_col = cfg['featurisation']['id_column']
    output_path = cfg['featurisation']['output_features_path']

    df = pd.read_csv(input_path)
    desc_df, failed = gen_custom_descriptors(df, smiles_col, id_col)

    desc_df.to_csv(output_path, index=False)

    print(f"Custom descriptors saved to {output_path}")
    if failed:
        print(f"Warning: failed to parse SMILES for IDs: {failed}")

if __name__ == "__main__":
    main()
