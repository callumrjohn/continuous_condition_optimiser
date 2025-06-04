import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold

# Split IDs into training and testing IDs using standard strategy - Split by unique ID (eg Substrate) only
def kfold_split(df, id_col, test_size = 0.2, shuffle = True, random_state = 42):
    unique_ids = df[id_col].unique()
    kf = KFold(n_splits=int(1/test_size), shuffle=shuffle, random_state=random_state)

    train_ids, test_ids = [], []
    for train_index, test_index in kf.split(unique_ids):
        train_ids.append(unique_ids[train_index])
        test_ids.append(unique_ids[test_index])

    return train_ids, test_ids


# Split IDs into training and testing IDs using Leave-One-Out (LOO) strategy - Split by unique ID (eg Substrate) only
def loo_split(df, id_col):

    unique_ids = df[id_col].unique()
    loo = LeaveOneOut()

    train_ids, test_ids = [], []
    for train_index, test_index in loo.split(unique_ids):
        train_ids.append(unique_ids[train_index])
        test_ids.append(unique_ids[test_index])

    return train_ids, test_ids


# Get training and testing DataFrames based on the split IDs
def get_test_train_df(df, train_ids, test_ids, id_col):
    
    train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    test_df = df[df[id_col].isin(test_ids)].reset_index(drop=True)

    return train_df, test_df
