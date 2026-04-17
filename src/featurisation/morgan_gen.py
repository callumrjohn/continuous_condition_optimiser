import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from src.utils.config import load_config


def gen_morgan_fps(df, smiles_col, id_col, radius=2, nBits=2048):
    """
    Generate Morgan fingerprints for molecules from SMILES strings.
    
    Args:
        df : pd.DataFrame
            Input DataFrame containing SMILES strings and molecule identifiers
        smiles_col : str
            Name of the column containing SMILES strings
        id_col : str
            Name of the column containing molecule identifiers
        radius : int, optional
            Radius of the Morgan fingerprint (default: 2)
        nBits : int, optional
            Number of bits in the fingerprint (default: 2048)
    
    Returns:
        tuple
            - fp_df (pd.DataFrame): DataFrame with Morgan fingerprint bits where the first
              column is id_col followed by 'bit_0' through 'bit_{nBits-1}' columns
            - failed (list): List of molecule IDs for which SMILES parsing failed
    
    Notes:
        Molecules with invalid SMILES strings have all fingerprint bits set to NaN.
        The fingerprint is converted to an integer numpy array where each element is
        either 0, 1, or NaN.
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)

    fps = []
    failed = []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            failed.append(row[id_col])
            fps.append(np.full(nBits, np.nan, dtype=float))
            continue
        
        # Generate fingerprint and convert to ExplicitBitVect for safer conversion
        fp = mfpgen.GetFingerprint(mol)
        # Use explicit bit vector conversion which is more stable than ConvertToNumpyArray
        fp_list = list(fp)
        fps.append(np.array(fp_list, dtype=int))

    # Assemble DataFrame with consistent data types
    fp_df = pd.DataFrame(fps, columns=[f'bit_{i}' for i in range(nBits)])
    fp_df[id_col] = df[id_col].values

    # Reorder columns
    cols = [id_col] + [f'bit_{i}' for i in range(nBits)]
    return fp_df[cols], failed


def main():
    """
    Generate Morgan fingerprints from input SMILES file and save to CSV.
    
    Loads configuration from YAML files, reads SMILES data from input CSV,
    generates Morgan fingerprints using gen_morgan_fps(), and saves the resulting
    fingerprint matrix to the specified output path. Issues warnings for any
    molecules with invalid SMILES strings.
    
    Configuration is loaded from 'configs/base.yaml' and 'configs/featurisation/morgan.yaml'
    which specify input/output paths and fingerprint parameters (radius and nBits).
    
    Returns:
        None
        Outputs a CSV file with Morgan fingerprints and prints the output path.
    """
    config_files = ["configs/base.yaml", "configs/featurisation/morgan.yaml"]
    cfg = load_config(config_files)

    input_path = cfg['featurisation']['smiles_path']
    smiles_col = cfg['featurisation']['smiles_column']
    id_col = cfg['featurisation']['id_column']
    output_path = cfg['featurisation']['output_features_path']

    radius = cfg['featurisation']['parameters'].get('radius', 2)
    nBits = cfg['featurisation']['parameters'].get('nBits', 2048)

    df = pd.read_csv(input_path)
    fp_df, failed = gen_morgan_fps(df, smiles_col, id_col, radius, nBits)

    fp_df.to_csv(output_path, index=False)

    print(f"Morgan fingerprints saved to {output_path}")
    if failed:
        print(f"Warning: failed to parse SMILES for IDs: {failed}")

if __name__ == "__main__":
    main()