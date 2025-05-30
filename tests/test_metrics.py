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