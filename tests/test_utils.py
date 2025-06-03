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



