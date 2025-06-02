import os
import sys
import pandas as pd
from glob import glob
from src.utils.config import load_config
from src.utils.load_model import load_model_class
from src.guis.data_model_selector import select_data_models_tkinter

def xy_split(df, remove_columns, target_columns):

    if isinstance(remove_columns, str):
        remove_columns = [remove_columns]
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    df = df.drop(columns=remove_columns)
    X = df.drop(columns=target_columns)
    y = df[target_columns] if len(target_columns) > 1 else df[target_columns[0]]
    return X, y

def main():
    config_files = ["configs/base.yaml", "configs/models/train_model.yaml"]
    cfg = load_config(config_files)

    input_data_dir = cfg['data']['model_input_dir']
    model_dir = cfg['model']['model_dir']

    data_names = [os.path.basename(f).replace('.csv', '') for f in glob(f'{input_data_dir}\*.csv')]
    model_names = [os.path.basename(f).replace('.py', '') for f in glob(f'{model_dir}\*.py')]

    training_dataset, model_name = select_data_models_tkinter(data_names, model_names)

    print(f"Selected Dataset: {training_dataset}, Model: {model_name}")
    #sys.exit()   

    # Update and reload configs based on selection
    config_files.append(f"configs/models/{model_name.lower()}.yaml")
    cfg = load_config(config_files)

    data_path = cfg['data']['model_input_dir'] + f"/{training_dataset}.csv"
    remove_columns = cfg['training']['columns_to_remove']
    target_columns = cfg['training']['target_variables']
    params = cfg['model']['parameters']

    # Load and split data
    df = pd.read_csv(data_path)

    X, y = xy_split(df, remove_columns, target_columns)

    # Load model class
    ModelClass = load_model_class(model_name)
    model = ModelClass(**params)
    model.train(X, y)

    # Save the model
    model_save_path = cfg['output']['output_model_dir'] + f"{model_name.lower()}_{training_dataset}.pkl"
    print(model_save_path)
    model.save(model_save_path)

    print(f"Model {model_name} trained and saved to {model_save_path}")

if __name__ == "__main__":
    main()