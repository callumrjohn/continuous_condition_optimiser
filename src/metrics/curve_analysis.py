import numpy as np


# Derive a yield curve from the given data points using linear interpolation.
def interpolate_data(X, y, inter_step = 0.1):
    
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


# Find the high yielding region based on a threshold of the maximum y value.
def find_region(X, y, threshold = 0.9):
    y = [x if x >= 0 else 0 for x in y]  # Ensure no negative values
    max_y = np.max(y)
    cutoff_y = max_y * threshold
    #print("Cutoff y value for optimum region:", cutoff_y)
    optimum_X_region = X[y >= cutoff_y]
    #print(len(optimum_X_region), "points in optimum region")
    opt_Xmin, opt_Xmax = np.min(optimum_X_region), np.max(optimum_X_region)
    return opt_Xmin, opt_Xmax


# Find the optimum point in the data, which is the point with the maximum y value.
def find_optimum(X, y):
    max_y = np.max(y)
    max_index = np.argmax(y)
    max_X = X[max_index]
    
    return max_X, max_y