import numpy as np


def test_interpolate_data():
    from src.metrics.curve_analysis import interpolate_data

    # Define test data
    X = [0, 1, 5]
    y = [0, 1, 30]

    # Interpolate data
    x_interpolated, y_interpolated = interpolate_data(X, y)

    # Check the length of the interpolated data
    assert len(x_interpolated) == 50
    assert len(y_interpolated) == 50

    # Check the first few values
    assert np.isclose(x_interpolated[0], 0.0)
    assert np.isclose(y_interpolated[0], 0.0)
    assert np.isclose(x_interpolated[-1], 5.0)
    assert np.isclose(y_interpolated[-1], 30.0)
    

def test_get_optimum():
    import pandas as pd
    from src.metrics.curve_analysis import interpolate_data, find_region, find_optimum

    # Make test df
    data = {
        'experiment': ['exp1', 'exp1', 'exp1', 'exp2', 'exp2', 'exp2'],
        'var': [0, 1, 2, 3, 4, 5],
        'value': [0, 1, 20, 50, 50, 20]
    }

    df = pd.DataFrame(data)

    # Define test data
    X = df['var'].values
    y = df['value'].values

    X_interpolated, y_interpolated = interpolate_data(X, y, inter_step=0.1)
    opt_Xmin, opt_Xmax = find_region(X_interpolated, y_interpolated, threshold=0.9)
    opt_X, opt_y = find_optimum(X_interpolated, y_interpolated)

    # Check the optimum values
    assert np.isclose(opt_Xmin, 2.88888)
    assert np.isclose(opt_Xmax, 4.11111)
    assert np.isclose(opt_X, 3.0)
    assert np.isclose(opt_y, 50.0)