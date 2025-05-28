import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit import Chem
from src.utils.config import load_config

def main():
    
    # Load configs
    config_files = ["configs/base.yaml", "configs/featurisation/rdkit.yaml"]
    cfg = load_config(config_files)

    input_path = cfg['featurisation']['smiles_path']
    smiles_col = cfg['featurisation']['smiles_column']
    id_col = cfg['featurisation']['id_column']
    
    output_path = cfg['featurisation']['output_features_path']

    # Parameters for RDKit fingerprint generation
    missingVal = cfg['featurisation']['parameters'].get('missingVal', np.NaN) # If a descriptor cannot be computed, use this value
    silent = cfg['featurisation']['parameters'].get('silent', True) # Don't print warnings for missing descriptors unless set to False

    # Read smiles CSV
    df = pd.read_csv(input_path)
    
    # Compute fingerprints
    rdk_disc = []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            mol = Chem.MolFromSmiles('')
        rdk_mol = Descriptors.CalcMolDescriptors(mol, missingVal=missingVal, silent=silent)
        rdk_disc.append(rdk_mol)


    # Convert to DataFrame
    rdk_df = pd.DataFrame(rdk_disc)
    rdk_df[id_col] = df[id_col]


    # Reorder columns: ID first
    cols = [id_col] + [col for col in rdk_df.columns if col != id_col]
    rdk_df = rdk_df[cols]


    # Save to CSV
    rdk_df.to_csv(output_path, index=False)


    print(f"RDKit descriptors saved to {output_path}")

if __name__ == "__main__":
    main()