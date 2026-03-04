import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from src.utils.config import load_config


def gen_morgan_fps(df, smiles_col, id_col, radius=2, nBits=2048):
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