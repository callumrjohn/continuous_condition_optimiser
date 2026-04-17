import numpy as np
import pandas as pd
from gauche.dataloader import MolPropLoader
from src.utils.config import load_config


def gen_fragprints(path, smiles_col, id_col):
    """
    Computes fragprints for moleculesspecified in a CSV file using the Gauche ML toolkit.
    
    Args:
        path : str
            Path to CSV file containing SMILES strings and molecule identifiers
        smiles_col : str
            Name of the column containing SMILES strings
        id_col : str
            Name of the column containing molecule identifiers
    
    Returns:
        pd.DataFrame
            DataFrame with fragprint bits where the first column is id_col,
            followed by 'bit_0' through 'bit_{nBits-1}' columns with binary values.
            Shape is (n_molecules, n_bits + 1)
    
    Notes:
        The Gauche MolPropLoader is used internally with validation disabled
        for processing molecules. Each bit represents the presence or absence
        of a particular molecular fragment or structural pattern.
    """

    df = pd.read_csv(path)
    loader = MolPropLoader()
    loader.read_csv(path, smiles_column=smiles_col, label_column=id_col, validate=False)
    loader.featurize('ecfp_fragprints')

    fragprints = loader.features
    nBits = fragprints.shape[1]

    # Assemble DataFrame
    fp_df = pd.DataFrame(fragprints, columns=[f'bit_{i}' for i in range(nBits)])
    fp_df[id_col] = df[id_col]

    # Reorder columns
    cols = [id_col] + [f'bit_{i}' for i in range(nBits)]
    fp_df = fp_df[cols]
    return fp_df[cols]


def main():
    """
    Generate fragprints from input SMILES file and save to CSV.
    
    Loads configuration from YAML files, reads SMILES data from input CSV,
    generates ECFp fragprints using gen_fragprints(), and saves the resulting
    fingerprint matrix to the specified output path.
    
    Configuration is loaded from 'configs/base.yaml' and 'configs/featurisation/fragprints.yaml'
    which specify input/output paths and SMILES/ID column names.
    
    Returns:
        None
        Outputs a CSV file with fragprints and prints the output path.
    """
    config_files = ["configs/base.yaml", "configs/featurisation/fragprints.yaml"]
    cfg = load_config(config_files)

    input_path = cfg['featurisation']['smiles_path']
    smiles_col = cfg['featurisation']['smiles_column']
    id_col = cfg['featurisation']['id_column']
    output_path = cfg['featurisation']['output_features_path']

    fp_df = gen_fragprints(input_path, smiles_col, id_col)

    fp_df.to_csv(output_path, index=False)

    print(f"Fragprints saved to {output_path}")

if __name__ == "__main__":
    main()