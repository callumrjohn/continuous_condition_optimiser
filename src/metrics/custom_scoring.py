# Calculate the accuracy of the predicted high yielding region. A high accuracy means the predicted midpoint is close to the actual midpoint of the region, and the predicted length is close to the actual length of the region.
def region_accuracy(Xmin, Xmax, X_predmax, X_predmin):
    midpoint = (Xmax + Xmin) / 2
    midpoint_pred = (X_predmax + X_predmin) / 2
    accuracy = 1 - abs(midpoint - midpoint_pred) / midpoint
    return accuracy


# Calculate the precision of the predicted high yielding region. A high precision means the type of response between curves is similar.
def region_precision(Xmin, Xmax, X_predmax, X_predmin):
    region_length = Xmax - Xmin
    region_length_pred = X_predmax - X_predmin
    precision = 1 - abs(region_length - region_length_pred) / region_length
    return precision


def region_overlap(Xmin, Xmax, X_predmin, X_predmax):
    intersection_min = max(Xmin, X_predmin)
    intersection_max = min(Xmax, X_predmax)
    overlap = max(0, intersection_max - intersection_min)
    return overlap


# Calculate the recall of the predicted high yielding region. A high recall means the predicted region is close to the actual region.
def region_recall(Xmin, Xmax, X_predmin, X_predmax):
    intersection_min = max(Xmin, X_predmin)
    intersection_max = min(Xmax, X_predmax)
    overlap = max(0, intersection_max - intersection_min)
    true_length = Xmax - Xmin
    recall = overlap / true_length if true_length > 0 else 0.0
    return recall


# Boolean checks to see if single condition is within the true region.
def is_midpoint_in_true_region(Xmin, Xmax, X_predmin, X_predmax):
    midpoint_pred = (X_predmax + X_predmin) / 2
    return Xmin <= midpoint_pred <= Xmax

def is_max_in_true_region(Xmin, Xmax, X_predopt):
    return Xmin <= X_predopt <= Xmax


# RUN THEM ALL TOGETHER
def run_custom_metrics(Xmin, Xmax, X_predmax, X_predmin, X_predopt):
    """
    Calculate custom metrics for the predicted high yielding region.
    """
    accuracy = region_accuracy(Xmin, Xmax, X_predmax, X_predmin)
    precision = region_precision(Xmin, Xmax, X_predmax, X_predmin)
    overlap = region_overlap(Xmin, Xmax, X_predmin, X_predmax)
    recall = region_recall(Xmin, Xmax, X_predmin, X_predmax)
    midpoint_in_true_region = is_midpoint_in_true_region(Xmin, Xmax, X_predmin, X_predmax)
    max_in_true_region = is_max_in_true_region(Xmin, Xmax, X_predopt)

    return accuracy, precision, overlap, recall, midpoint_in_true_region, max_in_true_region