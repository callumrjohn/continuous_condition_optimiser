import os
import importlib
from glob import glob
import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.guis.data_model_selector import select_data_models_tkinter
from src.guis.input_data_selector import select_input_data_tkinter


def load_model_class(model_name: str):

    config_files = ["configs/base.yaml"]
    cfg = load_config(config_files)

    model_dir = cfg['model']['model_dir']

    class_name = model_name[:-4].upper() + model_name[-4:] # asuming the class name is ACRONYMModel
    
    module_name = model_dir.replace('/', '.').replace('\\', '.')
    class_import = f"{module_name}{model_name.lower()}"
    try:
        module = importlib.import_module(class_import)
        model_class = getattr(module, class_name)
        return model_class
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not import {class_name} from {class_import}: {e}")


def select_model_and_data():

    config_files = ["configs/base.yaml"]
    cfg = load_config(config_files)

    input_data_dir = cfg['data']['model_input_dir']
    model_dir = cfg['model']['model_dir']
    model_config_dir = cfg['model']['model_config_dir']

    data_names = [os.path.basename(f).replace('.csv', '') for f in glob(f'{input_data_dir}\*.csv')]
    model_names = [os.path.basename(f).replace('.py', '') for f in glob(f'{model_dir}\*.py')]

    dset_name, model_name = select_data_models_tkinter(data_names, model_names)

    print(f"Selected Dataset: {dset_name}, Model: {model_name}")
    #sys.exit()   

    # Update and reload configs based on selection
    config_files.append(f"{model_config_dir}{model_name.lower()}.yaml")
    cfg = load_config(config_files)

    data_path = cfg['data']['model_input_dir'] + f"/{dset_name}.csv"
    params = cfg['model']['parameters']

    # Load model class
    ModelClass = load_model_class(model_name)
    model = ModelClass(**params)

    # Load and split data
    df = pd.read_csv(data_path)

    return model, df, model_name, dset_name

def select_input_data():

    config_files = ["configs/base.yaml"]
    cfg = load_config(config_files)

    input_data_dir = cfg['data']['model_input_dir']
    data_names = [os.path.basename(f).replace('.csv', '') for f in glob(f'{input_data_dir}\*.csv')]

    dset_name = select_input_data_tkinter(data_names)

    print(f"Selected Dataset: {dset_name}")

    data_path = os.path.join(input_data_dir, f"{dset_name}.csv")
    df = pd.read_csv(data_path)

    return df, dset_name


def xy_split(df, remove_columns, target_columns):

    if isinstance(remove_columns, str):
        remove_columns = [remove_columns]
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    df = df.drop(columns=remove_columns)
    X = df.drop(columns=target_columns)
    y = df[target_columns] # if len(target_columns) > 1 else df[target_columns[0]]
    return X.values, y.values

def extend_x(df, ind_var, remove_columns, target_columns, step = 0.2):

    if isinstance(remove_columns, str):
        remove_columns = [remove_columns]
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    if isinstance(ind_var, str):
        ind_var = [ind_var]
    if len(ind_var) > 1:
        raise ValueError("This function only supports a single independent variable.")    

    df = df.drop(columns=remove_columns)
    X = df.drop(columns=target_columns)
    ind_values = X[ind_var[0]].unique()  # Use ind_var[0] to get the column name as string
    granular_values = np.arange(ind_values.min(), ind_values.max(), step)
    other_cols = X.drop(ind_var, axis=1).drop_duplicates()
    granular_df = pd.DataFrame({ind_var[0]: granular_values})

    # Cross join to get all combinations
    X_expanded = other_cols.assign(key=1).merge(granular_df.assign(key=1), on='key').drop('key', axis=1)

    return X_expanded.values, X_expanded[ind_var[0]].values


def get_validation_dfs(df, id_col, splitter):
    unique_ids = df[id_col].unique()

    train_sets, test_sets = [], []
    for train_index, test_index in splitter.split(unique_ids):
        train_sets.append(unique_ids[train_index])
        test_sets.append(unique_ids[test_index])

    train_dfs = [df[df[id_col].isin(train_set)].reset_index(drop=True) for train_set in train_sets]
    test_dfs = [df[df[id_col].isin(test_set)].reset_index(drop=True) for test_set in test_sets]

    return train_dfs, test_dfs