import os
import importlib
from glob import glob
import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.guis.data_model_selector import select_data_models_tkinter
from src.guis.input_data_selector import select_input_data_tkinter


def load_model_class(model_name: str):
    """
    Dynamically load a model class by name from the configured model directory.
    
    Constructs the appropriate import path based on configuration and model name,
    then imports and returns the model class. Assumes class names follow the pattern
    ACRONYMModel (e.g., 'xgbmodel' -> 'XGBModel').
    
    Args:
        model_name: Name of the model module (e.g., 'xgbmodel', 'rfmodel')
    
    Returns:
        Model class (not instantiated)
    
    Raises:
        ImportError: If the module or class cannot be found
    
    Example:
        ModelClass = load_model_class('xgbmodel')
        model = ModelClass()
    """
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


def select_model_and_data(input_data_dir):
    """
    Launch an interactive GUI to select a model and dataset, then load both.
    
    Presents a tkinter GUI allowing users to select from available models and datasets.
    Loads the selected model class with its configuration and the selected dataset.
    
    Args:
        input_data_dir: Directory containing model input CSV files. If falsy, uses
            the directory from configuration.
    
    Returns:
        Tuple containing:
            - model: Instance of the selected model class with configured parameters
            - df: DataFrame containing the selected dataset
            - model_name: Name of the selected model
            - dset_name: Name of the selected dataset
    
    Example:
        model, df, model_name, dset_name = select_model_and_data(None)
    """
    config_files = ["configs/base.yaml"]
    cfg = load_config(config_files)

    if not input_data_dir:
        input_data_dir = cfg['data']['model_input_dir']
    model_dir = cfg['model']['model_dir']
    model_config_dir = cfg['model']['model_config_dir']

    data_names = [os.path.basename(f).replace('.csv', '') for f in glob(f'{input_data_dir}\*.csv')]
    model_names = [os.path.basename(f).replace('.py', '') for f in glob(f'{model_dir}\*.py') if not f.endswith('__init__.py')]

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
    """
    Launch an interactive GUI to select a dataset and load it.
    
    Presents a tkinter GUI allowing users to select from available model input datasets.
    Loads and returns the selected dataset as a DataFrame.
    
    Returns:
        Tuple containing:
            - df: DataFrame containing the selected dataset
            - dset_name: Name of the selected dataset
    
    Example:
        df, dset_name = select_input_data()
    """
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
    """
    Split a DataFrame into feature matrix X and target matrix y.
    
    Separates a DataFrame into independent variables (features) and dependent
    variables (targets), after removing specified columns. Both input parameters
    accept strings or lists.
    
    Args:
        df: Input DataFrame
        remove_columns: Column name(s) to remove (e.g., IDs). Can be string or list.
        target_columns: Column name(s) containing target variables. Can be string or list.
    
    Returns:
        Tuple containing:
            - X: NumPy array of features (shape: [n_samples, n_features])
            - y: NumPy array of targets (shape: [n_samples, n_targets])
    """
    if isinstance(remove_columns, str):
        remove_columns = [remove_columns]
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    df = df.drop(columns=remove_columns)
    X = df.drop(columns=target_columns)
    y = df[target_columns] # if len(target_columns) > 1 else df[target_columns[0]]
    return X.values, y.values

def extend_X(df, ind_var, remove_columns, target_columns, granular_values):
    """
    Generate expanded feature matrix with varying independent variable values.
    
    Creates an expanded feature matrix for prediction by varying a single independent
    variable across specified values while keeping all other features constant. Useful for
    generating prediction curves for sensitivity analysis.
    
    Args:
        df: Input DataFrame
        ind_var: Independent variable column name to vary (must be single column)
        remove_columns: Column name(s) to remove (e.g., IDs). Can be string or list.
        target_columns: Column name(s) containing target variables. Can be string or list.
        granular_values: List of values to vary the independent variable across
    
    Returns:
        Tuple containing:
            - X_expanded: NumPy array of repeated features with varying ind_var
            - ind_var_values: The granular_values as array
    
    Raises:
        ValueError: If multiple independent variables specified (function supports only one)
    """
    if isinstance(remove_columns, str):
        remove_columns = [remove_columns]
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    if isinstance(ind_var, str):
        ind_var = [ind_var]
    if len(ind_var) > 1:
        raise ValueError("This function only supports a single independent variable.")    

    # Drop ID and target columns (not part of features)
    df_features = df.drop(columns=remove_columns + target_columns)
    
    # Pick a representative row to clone all constant feature values
    base_row = df_features.iloc[0].copy()

    X_rows = []
    for val in granular_values:
        row = base_row.copy()
        row[ind_var[0]] = val
        X_rows.append(row)

    X_expanded = pd.DataFrame(X_rows)

    return X_expanded.values, X_expanded[ind_var[0]].values


def get_validation_dfs(df, id_col, splitter):
    """
    Generate train/test DataFrame splits using a cross-validation splitter.
    
    Uses a sklearn-like splitter object to create multiple train/test splits based on
    unique IDs (e.g., for leave-one-out or k-fold cross-validation). All rows with the
    same ID stay together in either train or test set.
    
    Args:
        df: Input DataFrame
        id_col: Column name containing ID values to split on
        splitter: Cross-validation splitter object with split(X) method
            (e.g., LeaveOneOut(), KFold(), etc. from sklearn.model_selection)
    
    Returns:
        Tuple containing:
            - train_dfs: List of training DataFrames (one per split)
            - test_dfs: List of test DataFrames (one per split)
    
    Example:
        from sklearn.model_selection import LeaveOneOut
        splitter = LeaveOneOut()
        train_dfs, test_dfs = get_validation_dfs(df, 'substrate_id', splitter)
    """
    unique_ids = df[id_col].unique()

    train_sets, test_sets = [], []
    for train_index, test_index in splitter.split(unique_ids):
        train_sets.append(unique_ids[train_index])
        test_sets.append(unique_ids[test_index])

    train_dfs = [df[df[id_col].isin(train_set)].reset_index(drop=True) for train_set in train_sets]
    test_dfs = [df[df[id_col].isin(test_set)].reset_index(drop=True) for test_set in test_sets]

    return train_dfs, test_dfs