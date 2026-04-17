import numpy as np


def interpolate_data(X, y, inter_step = 0.1):
    """
    Interpolate data points to create a smooth curve using linear interpolation.
    
    Creates a finely sampled response curve by linearly interpolating between
    the provided data points. The interpolated data is sorted and sampled at
    regular intervals specified by inter_step.
    
    Args:
        X : array-like
            Independent variable values (e.g., reaction temperature, catalyst load)
        y : array-like
            Dependent variable values (e.g., yield, conversion)
        inter_step : float, optional
            Interpolation step size (default: 0.1). Smaller values give finer
            interpolation but more compute time.
    
    Returns:
        tuple
            - X_interpolated (np.ndarray): Sorted and interpolated X values
            - y_interpolated (np.ndarray): Linearly interpolated y values
    """
    #Sort the data points based on X values
    X = np.array(X)
    y = np.array(y)

    sort_idx = np.argsort(X)
    X = X[sort_idx]
    y = y[sort_idx]

    X = np.array(X)
    y = np.array(y)

    X_interpolated, y_interpolated = [], []

    # Loop through each segment and interpolate linearly
    for i in range(len(X) - 1):
        n_steps = int((X[i + 1] - X[i]) / inter_step)
        x_segment = np.linspace(X[i], X[i + 1], n_steps)
        y_segment = np.linspace(y[i], y[i + 1], n_steps)
        
        X_interpolated.extend(x_segment)
        y_interpolated.extend(y_segment)
    
    X_interpolated = np.array(X_interpolated)
    y_interpolated = np.array(y_interpolated)

    return X_interpolated, y_interpolated


def find_region(X, y, threshold = 0.9):
    """
    Find the high-performing region of a response curve above a yield threshold.
    
    Identifies the range of independent variable values where the dependent variable
    (e.g., yield) exceeds a specified threshold relative to the maximum observed value.
    Useful for identifying experimental conditions that are "near-optimal".
    
    Args:
        X : array-like
            Independent variable values
        y : array-like
            Dependent variable values (typically yields or conversions)
        threshold : float, optional
            Fraction of maximum yield to use as cutoff (default: 0.9).
            For example, 0.9 means points where y >= 90% of maximum.
    
    Returns:
        tuple
            - opt_Xmin (float): Minimum X value in the high-performing region
            - opt_Xmax (float): Maximum X value in the high-performing region
    
    Notes:
        Negative y values are clipped to zero before threshold calculation.
        The region is defined by all X values where y >= threshold * max(y).
    """
    y = [x if x >= 0 else 0 for x in y]  # Ensure no negative values
    max_y = np.max(y)
    cutoff_y = max_y * threshold
    #print("Cutoff y value for optimum region:", cutoff_y)
    optimum_X_region = X[y >= cutoff_y]
    #print(len(optimum_X_region), "points in optimum region")
    opt_Xmin, opt_Xmax = np.min(optimum_X_region), np.max(optimum_X_region)
    return opt_Xmin, opt_Xmax


def find_optimum(X, y):
    """
    Find the optimum point in a response curve (point with maximum y value).
    
    Identifies the independent variable value and dependent variable value
    corresponding to the global maximum in the dataset.
    
    Args:
        X : array-like
            Independent variable values
        y : array-like
            Dependent variable values
    
    Returns:
        tuple
            - max_X (float): Independent variable value at the optimum
            - max_y (float): Maximum dependent variable value
    """
    max_y = np.max(y)
    max_index = np.argmax(y)
    max_X = X[max_index]
    
    return max_X, max_y