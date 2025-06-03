import os
import importlib
import glob
import pandas as pd
from src.utils.config import load_config
from src.guis.data_model_selector import select_data_models_tkinter

def load_model_class(model_name: str):
    module_name = model_name[:-4].upper() + model_name[-4:]
    module_path = f"src.models.architectures.{model_name.lower()}"
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, module_name)
        return model_class
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not import {module_name} from {module_path}: {e}")
    

def xy_split(df, remove_columns, target_columns):

    if isinstance(remove_columns, str):
        remove_columns = [remove_columns]
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    df = df.drop(columns=remove_columns)
    X = df.drop(columns=target_columns)
    y = df[target_columns] # if len(target_columns) > 1 else df[target_columns[0]]
    return X.values, y.values


def select_model_and_data():

    config_files = ["configs/base.yaml"]
    cfg = load_config(config_files)

    input_data_dir = cfg['data']['model_input_dir']
    model_dir = cfg['model']['model_dir']
    model_config_dir = cfg['model']['model_config_dir']

    data_names = [os.path.basename(f).replace('.csv', '') for f in glob(f'{input_data_dir}\*.csv')]
    model_names = [os.path.basename(f).replace('.py', '') for f in glob(f'{model_dir}\*.py')]

    training_dataset, model_name = select_data_models_tkinter(data_names, model_names)

    print(f"Selected Dataset: {training_dataset}, Model: {model_name}")
    #sys.exit()   

    # Update and reload configs based on selection
    config_files.append(f"{model_config_dir}{model_name.lower()}.yaml")
    cfg = load_config(config_files)

    data_path = cfg['data']['model_input_dir'] + f"/{training_dataset}.csv"
    params = cfg['model']['parameters']

    # Load model class
    ModelClass = load_model_class(model_name)
    model = ModelClass(**params)

    # Load and split data
    df = pd.read_csv(data_path)

    return model, df, model_name, training_dataset, cfg