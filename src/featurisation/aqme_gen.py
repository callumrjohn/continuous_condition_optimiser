import pandas as pd
from glob import glob
from aqme.csearch import csearch
from pathlib import Path
from aqme.qdescp import qdescp
from src.utils.config import load_config

def gen_aqme_descriptors(input_path, destination_dir, conformer_gen = 'rdkit', optimisation = 'xtb'):


    csearch(input=input_path, program=conformer_gen, output=destination_dir)

    csearch_dir = Path(destination_dir) / 'CSEARCH'
    conformer_files = [str(filepath) for filepath in csearch_dir.glob('*.sdf')]

    qdescp(files=conformer_files,
        program=optimisation,
        boltz=True,
        destination=destination_dir)

def main():
    config_files = ["configs/base.yaml", "configs/featurisation/aqme.yaml"]
    cfg = load_config(config_files)

    input_path = cfg['featurisation']['smiles_path']
    smiles_col = cfg['featurisation']['smiles_column']
    id_col = cfg['featurisation']['id_column']
    conformer_gen = cfg['featurisation']['parameters']['conformer_generation']
    optimisation = cfg['featurisation']['parameters']['optimiser']
    destination_dir = cfg['outputs']['aqme_output_dir']

    # Ensure destination directory exists
    Path(destination_dir).mkdir(parents=True, exist_ok=True)

    # Read input CSV file and rename columns for compatibility with AQME and export
    df = pd.read_csv(input_path)
    df_input = df[[smiles_col, id_col]]
    df_input = df_input.rename(columns={smiles_col: 'SMILES', id_col: 'code_name'})
    df_input_path = Path(destination_dir) / 'smiles_aqme.csv'
    df_input.to_csv(df_input_path, index=False)

    gen_aqme_descriptors(df_input_path, destination_dir, conformer_gen, optimisation)
    print(f"Raw AQME descriptors generated and saved to {destination_dir}")

if __name__ == "__main__":
    main()