import os
import sys
import pandas as pd
from glob import glob
from src.utils.config import load_config
from src.utils.model_utils import xy_split, select_model_and_data
from src.guis.data_model_selector import select_data_models_tkinter

def main():
    """
    Main entry point for training a machine learning model.
    
    Launches an interactive GUI for users to select a model and dataset combination.
    Trains the selected model on the entire dataset and saves it for later use
    in predictions or validation.
    
    Returns:
        None (saves trained model to file)
    """
    # Generate model, df of training data plus model name, training dataset name and relevant config
    model, df, model_name, dset_name = select_model_and_data()

    # Load the configuration file
    config_files = ["configs/base.yaml", "configs/models/train_model.yaml"]
    cfg = load_config(config_files)

    remove_columns = cfg['training']['columns_to_remove']
    target_columns = cfg['training']['target_variables']

    # Seperate the model into features and target variables and train...
    X, y = xy_split(df, remove_columns, target_columns)
    model.train(X, y)

    # Save the model for later use
    model_save_path = cfg['output']['output_model_dir'] + f"{model_name.lower()}_{dset_name}.pkl"
    print(model_save_path)
    model.save(model_save_path)

    print(f"Model {model_name} trained and saved to {model_save_path}")

if __name__ == "__main__":
    main()