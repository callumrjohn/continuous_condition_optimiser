import numpy as np
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from src.utils.config import load_config
import warnings

warnings.filterwarnings("ignore")

def gen_mordred_descriptors(df, smiles_col, id_col, ignore_3D = True, missingVal = None):
    """
    Generate Mordred molecular descriptors for molecules from SMILES strings.
    
    Args:
        df : pd.DataFrame
            Input DataFrame containing SMILES strings and molecule identifiers
        smiles_col : str
            Name of the column containing SMILES strings
        id_col : str
            Name of the column containing molecule identifiers
        ignore_3D : bool, optional
            If True, skip calculation of 3D descriptors (default: True)
        missingVal : float or None, optional
            Value to use for invalid SMILES or missing descriptors (default: None)
    
    Returns:
        pd.DataFrame
            DataFrame with computed Mordred descriptors where the first column is id_col,
            followed by all descriptor columns. Shape is (n_molecules, n_descriptors + 1)
    
    Notes:
        Invalid SMILES and non-numeric descriptor values are handled gracefully.
        Warnings are suppressed during descriptor calculation to avoid verbosity.
        The output maintains the same row order as the input DataFrame.
    """

    calc = Calculator(descriptors, ignore_3D=ignore_3D)
    desc_names = [str(desc) for desc in calc.descriptors]
    #print(desc_names[:10])
    mord_disc = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            # Invalid SMILES: set all descriptors to missingVal
            mord_mol = [missingVal] * len(desc_names)
        else:
            mord_mol = calc(mol)
            # Convert all non-integer/non-float values to missingVal
            mord_mol = [missingVal if not isinstance(val, (int, float)) else val for val in mord_mol]
        mord_disc.append(mord_mol)

    # Assemble DataFrame
    mord_df = pd.DataFrame(mord_disc, columns=desc_names)
    mord_df[id_col] = df[id_col]

    # Reorder columns
    cols = [id_col] + desc_names

    return mord_df[cols]


def main():
    """
    Generate Mordred descriptors from input SMILES file and save to CSV.
    
    Loads configuration from YAML files, reads SMILES data from input CSV,
    generates Mordred descriptors using gen_mordred_descriptors(), and saves
    the resulting feature matrix to the specified output path.
    
    Configuration is loaded from 'configs/base.yaml' and 'configs/featurisation/mordred.yaml'
    which specify input/output paths and descriptor generation parameters (ignore_3D,
    missingVal).
    
    Returns:
        None
        Outputs a CSV file with Mordred descriptors and prints the output path.
    """
    
    # Load configs
    config_files = ["configs/base.yaml", "configs/featurisation/mordred.yaml"]
    cfg = load_config(config_files)

    input_path = cfg['featurisation']['smiles_path']
    smiles_col = cfg['featurisation']['smiles_column']
    id_col = cfg['featurisation']['id_column']
    
    output_path = cfg['featurisation']['output_features_path']

    # Parameters for RDKit fingerprint generation
    missingVal = cfg['featurisation']['parameters']['missingVal'] # Value to use for missing descriptors
    ignore_3D = cfg['featurisation']['parameters']['ignore_3D'] # If a descriptor cannot be computed, use this value

    # Read smiles CSV
    df = pd.read_csv(input_path)
    mord_df = gen_mordred_descriptors(df, smiles_col, id_col, ignore_3D=ignore_3D, missingVal=missingVal)

    mord_df.to_csv(output_path, index=False)

    print(f"Mordred descriptors saved to {output_path}")

if __name__ == "__main__":
    main()