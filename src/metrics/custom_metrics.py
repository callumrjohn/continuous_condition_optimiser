# Calculate the accuracy of the predicted high yielding region. A high accuracy means the predicted midpoint is close to the actual midpoint of the region, and the predicted length is close to the actual length of the region.
def region_accuracy(Xmin, Xmax, X_predmin, X_predmax, scaler_min, scaler_max):
    """Value between 0 and 1 describing how close the midpoints of the regions are."""
    midpoint_true = (Xmax + Xmin) / 2
    midpoint_pred = (X_predmax + X_predmin) / 2
    max_dist = scaler_max - scaler_min
    accuracy = 1 - abs(midpoint_true - midpoint_pred) / max_dist if max_dist > 0 else 0.0
    return accuracy


# Calculate the precision of the predicted high yielding region. A high precision means the type of response between curves is similar.
def region_precision(Xmin, Xmax, X_predmin, X_predmax, scaler_min, scaler_max):
    """Value between 0 and 1 describing how similar the regions are in terms of width."""
    width_true = abs(Xmax - Xmin)
    width_pred = abs(X_predmax - X_predmin)
    max_width = abs(scaler_max - scaler_min)
    precision = 1 - abs(width_true - width_pred) / max_width if max_width > 0 else 0.0
    return precision


def region_overlap(Xmin, Xmax, X_predmin, X_predmax):
    """Value between 0 and 1 describing how closely the regions overlap (Jaccard index)."""
    intersection = max(0, min(Xmax, X_predmax) - max(Xmin, X_predmin))
    pred_length = abs(X_predmax - X_predmin)
    region_overlap = intersection / pred_length
    return region_overlap


# Calculate the recall of the predicted high yielding region. A high recall means the predicted region is close to the actual region.
def region_recall(Xmin, Xmax, X_predmin, X_predmax):
    """Value between 0 and 1: (length of overlap) / (length of true region)."""
    intersection = max(0, min(Xmax, X_predmax) - max(Xmin, X_predmin))
    true_length = abs(Xmax - Xmin)
    recall = intersection / true_length
    return recall


# Boolean checks to see if single condition is within the true region.
def is_midpoint_in_true_region(Xmin, Xmax, X_predmin, X_predmax):
    midpoint_pred = (X_predmax + X_predmin) / 2
    return Xmin <= midpoint_pred <= Xmax

def is_max_in_true_region(Xmin, Xmax, X_predopt):
    return Xmin <= X_predopt <= Xmax