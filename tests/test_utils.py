import numpy as np
import pandas as pd

def test_xy_split():
    from src.models.train_model import xy_split

    # Create a mock DataFrame
    df = pd.DataFrame({
        'id': ['sample1', 'sample2', 'sample3'],
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [0, 1, 0],
        'target': [70, 80, 90]
    })

    remove_columns = 'id'
    target_columns = 'target'

    # Split the DataFrame
    X, y = xy_split(df, remove_columns, target_columns)

    # Check the shape of X and y
    assert X.shape == (3, 3)  # One feature remaining
    assert y.shape == (3, 1)     # One target variable

    # Check the values in X and y
    expected_X = np.array([
        [1, 4, 0],
        [2, 5, 1],
        [3, 6, 0]
    ])
    expected_y = np.array([[70, 80, 90]]).reshape(-1, 1)
    
    assert np.array_equal(X, expected_X)
    assert np.array_equal(y, expected_y)

def test_extend_x():
    from src.utils.model_utils import extend_x

    # Create a mock DataFrame
    df = pd.DataFrame({
        'id': ['sample1', 'sample2', 'sample3'],
        'feature1': [1, 1, 1],
        'feature2': [3, 3, 3],
        'feature3': [2, 2, 2],
        'feature4': [0, 3, 4],
        'target': [70, 80, 90]
    })

    ind_var = 'feature4'
    remove_columns = ['id']
    target_columns = 'target'
    # Extend the DataFrame
    X_extended, X_extended_values = extend_x(df, ind_var, remove_columns, target_columns, step=0.1)
    # Check the shape of the extended DataFrame
    assert X_extended[:, 3].tolist() == np.arange(0, 4, 0.1).tolist()
    assert X_extended_values == np.arange(0, 4, 0.1).tolist()
    assert len(set(X_extended[:, 2])) == 1
    assert X_extended.shape == (40, 4)
    

def test_get_validation_df_kfold():
    from src.utils.model_utils import get_validation_dfs
    from sklearn.model_selection import KFold

    # Create a mock DataFrame
    df = pd.DataFrame({
        'id': ['sample1', 'sample2', 'sample3', 'sample4'],
        'feature1': [1, 2, 3, 4],
        'target': [10, 20, 30, 40]
    })

    id_col = 'id'
    splitter = KFold(n_splits=2)

    train_dfs, test_dfs = get_validation_dfs(df, id_col, splitter)

    # Check the number of train and test sets
    assert len(train_dfs) == 2
    assert len(test_dfs) == 2

    # Check the shapes of the first train and test DataFrames
    assert train_dfs[0].shape == (2, 3)  # Two samples in train set
    assert test_dfs[0].shape == (2, 3)   # Two samples in test set

    # Check the IDs in the first train and test DataFrames
    assert set(train_dfs[0][id_col]) == {'sample3', 'sample4'}
    assert set(test_dfs[0][id_col]) == {'sample1', 'sample2'}

def test_get_validation_df_loo():
    from src.utils.model_utils import get_validation_dfs
    from sklearn.model_selection import LeaveOneOut

    df = pd.DataFrame({
        'id': ['sample1', 'sample2', 'sample3', 'sample4'],
        'feature1': [1, 2, 3, 4],
        'target': [10, 20, 30, 40]
    })

    id_col = 'id'
    splitter = LeaveOneOut()

    train_dfs, test_dfs = get_validation_dfs(df, id_col, splitter)

    # Check the number of train and test sets
    assert len(train_dfs) == 4
    assert len(test_dfs) == 4

    # Check the shapes of the first train and test DataFrames
    assert train_dfs[0].shape == (3, 3)  # Three samples in train set
    assert test_dfs[0].shape == (1, 3)   # One sample in test set

    # Check the IDs in the first train and test DataFrames
    assert set(train_dfs[0][id_col]) == {'sample2', 'sample3', 'sample4'}
    assert set(test_dfs[0][id_col]) == {'sample1'}