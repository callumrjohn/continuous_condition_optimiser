import numpy as np
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit import Chem
from src.utils.config import load_config


def main():
    
    # Load configs
    config_files = ["configs/base.yaml", "configs/featurisation/morgan.yaml"]
    cfg = load_config(config_files)

    input_path = cfg['featurisation']['smiles_path']
    smiles_col = cfg['featurisation']['smiles_column']
    id_col = cfg['featurisation']['id_column']
    
    output_path = cfg['featurisation']['output_features_path']



    # Initialise Morgan fingerprint generator with parameters
    radius = cfg['featurisation']['parameters'].get('radius', 2) # Default radius for Morgan fingerprints if not specified
    nBits = cfg['featurisation']['parameters'].get('nBits', 2048) # Default number of bits for Morgan fingerprints if not specified
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)

    # Read smiles CSV
    df = pd.read_csv(input_path)
    

    # Compute fingerprints
    fps = []
    failed = []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            failed.append(row[id_col])
            fps.append([np.nan] * nBits)
            continue
        fp = mfpgen.GetFingerprint(mol)
        fp_arr = np.zeros((nBits,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
        fps.append(fp_arr)


    # Convert to DataFrame
    fp_df = pd.DataFrame(fps, columns=[f'bit_{i}' for i in range(nBits)])
    fp_df[id_col] = df[id_col]


    # Reorder columns: ID first
    cols = [id_col] + [f'bit_{i}' for i in range(nBits)]
    fp_df = fp_df[cols]


    # Save to CSV
    fp_df.to_csv(output_path, index=False)


    print(f"Morgan fingerprints saved to {output_path}")
    if failed:
        print(f"Warning: failed to parse SMILES for IDs: {failed}")

if __name__ == "__main__":
    main()