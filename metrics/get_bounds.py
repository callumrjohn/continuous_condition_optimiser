import numpy as np

def find_region(X, y, threshold = 0.8, inter_step = 0.1):
        

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    x_interpolated, y_interpolated = [], []

    # Loop through each segment and interpolate linearly
    for i in range(len(X) - 1):
        x_segment = np.arange(X[i], X[i + 1], inter_step)
        y_segment = np.arange(y[i], y[i + 1], inter_step)
        
        x_interpolated.extend(x_segment)
        y_interpolated.extend(y_segment)
    
    x_interpolated = np.array(x_interpolated)
    y_interpolated = np.array(y_interpolated)

    # Calculate max y value and find corresponding x value
    max_y = np.max(y_interpolated)
    max_y_index = np.argmax(y_interpolated)

    # Find high yielding region
    cutoff_y = max_y * threshold
    optimum_x_region = x_interpolated[y_interpolated >= cutoff_y]
    opt_min, opt_max = np.min(optimum_x_region), np.max(optimum_x_region)


    return opt_min, opt_max
