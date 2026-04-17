import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold

def kfold_split(df, id_col, test_size = 0.2, shuffle = True, random_state = 42):
    """
    Split DataFrame IDs into k-fold train/test sets for cross-validation.
    
    Performs k-fold cross-validation splitting based on unique IDs (e.g., substrates),
    ensuring all rows with the same ID stay together in either train or test set.
    
    Args:
        df: Input DataFrame
        id_col: Column name containing unique identifiers to split on
        test_size: Fraction of data for test set (default: 0.2 for 5-fold CV).
            Number of folds = 1/test_size
        shuffle: Whether to shuffle before splitting (default: True)
        random_state: Random seed for reproducible splits (default: 42)
    
    Returns:
        Tuple containing:
            - train_ids: List of train ID arrays (one per fold)
            - test_ids: List of test ID arrays (one per fold)
    """
    unique_ids = df[id_col].unique()
    kf = KFold(n_splits=int(1/test_size), shuffle=shuffle, random_state=random_state)

    train_ids, test_ids = [], []
    for train_index, test_index in kf.split(unique_ids):
        train_ids.append(unique_ids[train_index])
        test_ids.append(unique_ids[test_index])

    return train_ids, test_ids


def loo_split(df, id_col):
    """
    Split DataFrame IDs into leave-one-out (LOO) cross-validation sets.
    
    Performs leave-one-out cross-validation splitting based on unique IDs,
    creating n splits where n is the number of unique IDs. Each split uses
    one ID for testing and remaining IDs for training.
    
    Args:
        df: Input DataFrame
        id_col: Column name containing unique identifiers to split on
    
    Returns:
        Tuple containing:
            - train_ids: List of train ID arrays (one per split)
            - test_ids: List of test ID arrays (one per split, single ID each)
    """
    unique_ids = df[id_col].unique()
    loo = LeaveOneOut()

    train_ids, test_ids = [], []
    for train_index, test_index in loo.split(unique_ids):
        train_ids.append(unique_ids[train_index])
        test_ids.append(unique_ids[test_index])

    return train_ids, test_ids


def get_test_train_df(df, train_ids, test_ids, id_col):
    """
    Extract train and test DataFrames based on specified ID sets.
    
    Filters the DataFrame into train and test subsets using provided sets of IDs,
    keeping all rows matching each ID together.
    
    Args:
        df: Input DataFrame
        train_ids: Array or list of IDs for training set
        test_ids: Array or list of IDs for test set
        id_col: Column name containing IDs
    
    Returns:
        Tuple containing:
            - train_df: DataFrame containing rows where id_col is in train_ids
            - test_df: DataFrame containing rows where id_col is in test_ids
    """
    train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    test_df = df[df[id_col].isin(test_ids)].reset_index(drop=True)

    return train_df, test_df
