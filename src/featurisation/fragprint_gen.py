import numpy as np
import pandas as pd
from gauche.dataloader import MolPropLoader
from src.utils.config import load_config


def gen_fragprints(path, smiles_col, id_col):

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