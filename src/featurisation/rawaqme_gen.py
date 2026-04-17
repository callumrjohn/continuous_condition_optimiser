import pandas as pd
from glob import glob
from aqme.csearch import csearch
from pathlib import Path
from aqme.qdescp import qdescp
from src.utils.config import load_config

def gen_aqme_descriptors(input_path, destination_dir, conformer_gen = 'rdkit', optimisation = 'xtb'):
    """
    Generate AQME descriptors through conformer search and quantum mechanical optimization.
    
    Orchestrates the AQME pipeline for descriptor generation, which includes: (1) conformer
    generation using specified method to create 3D structures from SMILES, (2) optimization
    of generated conformers using specified quantum mechanical or semi-empirical method.
    Outputs are organized in subdirectories (CSEARCH for conformers, QDESCP for descriptors).
    
    Args:
        input_path : str
            Path to CSV file with SMILES strings and molecule identifiers in columns
            named 'SMILES' and 'code_name' respectively
        destination_dir : str
            Directory where output files and subdirectories will be created
        conformer_gen : str, optional
            Method for conformer generation (default: 'rdkit'). Common options:
            'rdkit', 'distance_geometry', etc.
        optimisation : str, optional
            Semi-empirical or quantum mechanical method for optimization (default: 'xtb').
            Must be installed separately (XTB, MOPAC, etc.)
    
    Returns:
        None
        Outputs conformer files in destination_dir/CSEARCH/ and descriptor files in
        destination_dir/QDESCP/
    
    Notes:
        This is a computationally expensive function that may take significant time
        depending on the number of molecules and method chosen. Requires external
        dependencies (XTB or similar installed).
    """


    csearch(input=input_path, program=conformer_gen, output=destination_dir)

    csearch_dir = Path(destination_dir) / 'CSEARCH'
    conformer_files = [str(filepath) for filepath in csearch_dir.glob('*.sdf')]

    qdescp(files=conformer_files,
        program=optimisation,
        boltz=True,
        destination=destination_dir)

def main():
    """
    Generate raw AQME descriptors from input SMILES file through full optimization pipeline.
    
    Loads configuration from YAML files, reads SMILES data from input CSV, validates
    output directory, prepares formatted input for AQME, and calls gen_aqme_descriptors()
    to generate conformers and compute quantum mechanical descriptors.
    
    The pipeline includes conformer generation and geometry optimization, producing
    both structural files (.sdf) and computed molecular descriptors.
    
    Configuration is loaded from 'configs/base.yaml' and 'configs/featurisation/aqme.yaml'
    which specify input/output paths, column names, and AQME parameters (conformer
    generation method, optimization method/software).
    
    Returns:
        None
        Outputs conformer directories and descriptor files in the AQME output directory
        and prints confirmation message.
    
    Notes:
        Creates intermediate 'smiles_aqme.csv' file in the output directory with
        renamed columns compatible with AQME tools (SMILES, code_name).
    """
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