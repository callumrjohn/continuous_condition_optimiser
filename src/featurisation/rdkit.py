import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit import Chem
from src.utils.config import load_config


def gen_rdkit_descriptors(df, smiles_col, id_col, missingVal=np.nan, silent=True):
    desc_names = [desc[0] for desc in Descriptors._descList]
    rdk_disc = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            # Invalid SMILES: set all descriptors to missingVal
            rdk_mol = dict(zip(desc_names, [missingVal] * len(desc_names)))
        else:
            rdk_mol = Descriptors.CalcMolDescriptors(mol, missingVal=missingVal, silent=silent)
        rdk_disc.append(rdk_mol)

    # Assemble DataFrame
    rdk_df = pd.DataFrame(rdk_disc)
    rdk_df[id_col] = df[id_col]

    # Reorder columns
    cols = [id_col] + [col for col in rdk_df.columns if col != id_col]
    return rdk_df[cols]


def main():
    
    # Load configs
    config_files = ["configs/base.yaml", "configs/featurisation/rdkit.yaml"]
    cfg = load_config(config_files)

    input_path = cfg['featurisation']['smiles_path']
    smiles_col = cfg['featurisation']['smiles_column']
    id_col = cfg['featurisation']['id_column']
    
    output_path = cfg['featurisation']['output_features_path']

    # Parameters for RDKit fingerprint generation
    missingVal = cfg['featurisation']['parameters'].get('missingVal', np.nan) # If a descriptor cannot be computed, use this value
    silent = cfg['featurisation']['parameters'].get('silent', True) # Don't print warnings for missing descriptors unless set to False

    # Read smiles CSV
    df = pd.read_csv(input_path)
    rdk_df = gen_rdkit_descriptors(df, smiles_col, id_col, missingVal=missingVal, silent=silent)

    rdk_df.to_csv(output_path, index=False)

    print(f"RDKit descriptors saved to {output_path}")

if __name__ == "__main__":
    main()