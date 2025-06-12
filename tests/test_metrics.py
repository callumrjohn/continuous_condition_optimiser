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

def test_region_accuracy():
    from src.metrics.custom_metrics import region_accuracy

    # Define test data
    Xmins = [0, 0, 0]
    Xmaxs = [10, 10, 0]
    X_predmins = [2, 4, 10] 
    X_predmaxs = [6, 6, 10]
    
    scaler_min = 0
    scaler_max = 10
    
    scores = []
    expected_scores = [0.9, 1.0, 0.0]  # Expected accuracy values for the test cases
    
    for Xmin, Xmax, X_predmin, X_predmax in zip(Xmins, Xmaxs, X_predmins, X_predmaxs):
        accuracy = region_accuracy(Xmin, Xmax, X_predmin, X_predmax, scaler_min, scaler_max)
        scores.append(accuracy)

    # Check the accuracy values
    for score, expected in zip(scores, expected_scores):
        assert np.isclose(score, expected)


def test_region_precision():
    from src.metrics.custom_metrics import region_precision

    # Define test data
    Xmins = [0, 0, 0]
    Xmaxs = [10, 10, 0]
    X_predmins = [2, 4, 10] 
    X_predmaxs = [8, 6, 10]
    
    scaler_min = 0
    scaler_max = 10
    
    scores = []
    expected_scores = [0.6, 0.2, 1.0]  # Expected precision values for the test cases
    
    for Xmin, Xmax, X_predmin, X_predmax in zip(Xmins, Xmaxs, X_predmins, X_predmaxs):
        precision = region_precision(Xmin, Xmax, X_predmin, X_predmax, scaler_min, scaler_max)
        scores.append(precision)

    # Check the precision values
    for score, expected in zip(scores, expected_scores):
        assert np.isclose(score, expected)


def test_region_overlap():
    from src.metrics.custom_metrics import region_overlap

    # Define test data
    Xmins = [0, 2, 1, 4]
    Xmaxs = [10, 6, 5, 6]
    X_predmins = [2, 6, 4, 1] 
    X_predmaxs = [8, 8, 6, 5]
    
    scores = []
    expected_scores = [1.0, 0.0, 0.5, 0.25]  # Expected overlap values for the test cases
    
    for Xmin, Xmax, X_predmin, X_predmax in zip(Xmins, Xmaxs, X_predmins, X_predmaxs):
        overlap = region_overlap(Xmin, Xmax, X_predmin, X_predmax)
        scores.append(overlap)

    # Check the overlap values
    for score, expected in zip(scores, expected_scores):
        assert np.isclose(score, expected)


def test_region_recall():
    from src.metrics.custom_metrics import region_recall

    # Define test data
    Xmins = [0, 2, 1, 4]
    Xmaxs = [10, 6, 5, 6]
    X_predmins = [2, 6, 4, 1] 
    X_predmaxs = [8, 8, 6, 6]
    
    scores = []
    expected_scores = [0.6, 0.0, 0.25, 1.0]  # Expected recall values for the test cases
    
    for Xmin, Xmax, X_predmin, X_predmax in zip(Xmins, Xmaxs, X_predmins, X_predmaxs):
        recall = region_recall(Xmin, Xmax, X_predmin, X_predmax)
        scores.append(recall)

    # Check the recall values
    for score, expected in zip(scores, expected_scores):
        assert np.isclose(score, expected)


def test_is_midpoint_in_true_region():
    from src.metrics.custom_metrics import is_midpoint_in_true_region

    # Define test data
    Xmins = [0, 2, 1, 4]
    Xmaxs = [10, 6, 5, 6]
    X_predmins = [2, 6, 4, 1] 
    X_predmaxs = [8, 8, 6, 5]
    
    expected_results = [True, False, True, False]  # Expected boolean results for the test cases
    
    results = []
    for Xmin, Xmax, X_predmin, X_predmax in zip(Xmins, Xmaxs, X_predmins, X_predmaxs):
        result = is_midpoint_in_true_region(Xmin, Xmax, X_predmin, X_predmax)
        results.append(result)

    # Check the boolean results
    for result, expected in zip(results, expected_results):
        assert result == expected


def test_is_max_in_true_region():
    from src.metrics.custom_metrics import is_max_in_true_region

    # Define test data
    Xmins = [0, 2, 1, 4]
    Xmaxs = [10, 6, 5, 6]
    X_predopts = [8, 8, 6, 5]  # Predicted optima
    
    expected_results = [True, False, False, True]  # Expected boolean results for the test cases
    
    results = []
    for Xmin, Xmax, X_predopt in zip(Xmins, Xmaxs, X_predopts):
        result = is_max_in_true_region(Xmin, Xmax, X_predopt)
        results.append(result)

    # Check the boolean results
    for result, expected in zip(results, expected_results):
        assert result == expected