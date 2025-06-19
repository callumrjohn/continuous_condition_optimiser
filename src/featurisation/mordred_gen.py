import numpy as np
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from src.utils.config import load_config
import warnings

warnings.filterwarnings("ignore")

def gen_mordred_descriptors(df, smiles_col, id_col, ignore_3D = True, missingVal = np.nan):

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
        mord_disc.append(mord_mol)

    # Assemble DataFrame
    mord_df = pd.DataFrame(mord_disc, columns=desc_names)
    mord_df[id_col] = df[id_col]

    # Reorder columns
    cols = [id_col] + desc_names

    return mord_df[cols]


def main():
    
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