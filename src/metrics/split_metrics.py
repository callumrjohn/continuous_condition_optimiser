import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from src.utils.data_utils import yield_to_unbounded, unbounded_to_yield
from src.utils.model_utils import xy_split, extend_X
from src.metrics.curve_analysis import interpolate_data, find_region, find_optimum
from src.metrics.custom_metrics import region_accuracy, region_precision, region_overlap, region_recall, is_midpoint_in_true_region, is_max_in_true_region


def run_custom_metrics(Xmin, Xmax, X_predmax, X_predmin, X_predopt, scaler_min=0, scaler_max=25):
    """
    Calculate region-based custom metrics comparing predicted and experimental optimum regions.
    
    Computes six custom metrics that evaluate the model's ability to correctly identify
    the high-yielding experimental region by comparing predicted region boundaries with
    experimental optimum region boundaries.
    
    Args:
        Xmin : float
            Minimum boundary of experimental optimum region
        Xmax : float
            Maximum boundary of experimental optimum region
        X_predmax : float
            Maximum boundary of predicted optimum region
        X_predmin : float
            Minimum boundary of predicted optimum region
        X_predopt : float
            Independent variable value at predicted optimum
        scaler_min : float, optional
            Minimum value in observed data range (default: 0)
        scaler_max : float, optional
            Maximum value in observed data range (default: 25)
    
    Returns:
        tuple
            Six-element tuple containing:
            - accuracy (float): Fraction of experimental region correctly predicted
            - precision (float): Fraction of predicted region within experimental region
            - overlap (float): Overlap between regions as fraction of union
            - recall (float): Sensitivity of prediction to experimental region
            - midpoint_in_true_region (bool): Whether predicted optimum midpoint is in true region
            - max_in_true_region (bool): Whether predicted optimum maximum is in true region
    
    Notes:
        All metrics scale the regions to [scaler_min, scaler_max] range for comparison.
        Region accuracy measures what fraction of the true region was predicted.
        Precision measures what fraction of predicted region is correct.
    """
    accuracy = region_accuracy(Xmin, Xmax, X_predmax, X_predmin, scaler_min=scaler_min, scaler_max=scaler_max)
    precision = region_precision(Xmin, Xmax, X_predmax, X_predmin, scaler_min=scaler_min, scaler_max=scaler_max)
    overlap = region_overlap(Xmin, Xmax, X_predmin, X_predmax)
    recall = region_recall(Xmin, Xmax, X_predmin, X_predmax)
    midpoint_in_true_region = is_midpoint_in_true_region(Xmin, Xmax, X_predmin, X_predmax)
    max_in_true_region = is_max_in_true_region(Xmin, Xmax, X_predopt)

    return accuracy, precision, overlap, recall, midpoint_in_true_region, max_in_true_region


def evaluate_split_standard(model,
                   id_var,
                   dep_vars,
                   train_df,
                   test_df,
                   sigmoid_bound=False):
    """
    Evaluate model performance on a train/test split using standard regression metrics.
    
    Trains a model on the training set and computes standard regression metrics
    (MSE, MAE, R²) on the test set. Includes optional yield transformation using
    sigmoid bounding for models that predict on unbounded scale.
    
    Args:
        model : Model class
            Model instance with train() and predict() methods
        id_var : str
            Column name containing sample identifiers to remove before training
        dep_vars : str or list
            Column name(s) of dependent variables (targets) to predict
        train_df : pd.DataFrame
            Training data DataFrame
        test_df : pd.DataFrame
            Test data DataFrame
        sigmoid_bound : bool, optional
            If True, applies logit transformation to y_train and inverse logit
            to y_pred to bound predictions to [0, 100] (default: False)
    
    Returns:
        dict
            Dictionary with keys:
            - 'mae': Mean Absolute Error
            - 'mse': Mean Squared Error
            - 'r2score': R² score
    
    Notes:
        Features are standardized using StandardScaler. The model is trained once
        and evaluated on the test set.
    """
    X_train, y_train = xy_split(train_df, id_var, dep_vars)
    X_test, y_test = xy_split(test_df, id_var, dep_vars)

    # Train the model
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if sigmoid_bound:
        y_train = yield_to_unbounded(y_train)

    model.train(X_train, y_train)

    # Calculate standard metrics
    y_pred = model.predict(X_test)
    
    if sigmoid_bound:
        y_pred = unbounded_to_yield(y_pred)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2score = r2_score(y_test, y_pred)

    return {'mae': mae, 'mse': mse, 'r2score': r2score}


def evaluate_split_custom(model,
                   id_var,
                   dep_var,
                   indep_var,
                   constant_vars,
                   df_train,
                   df_test,
                   df_exp_optimum,
                   iter_step=0.1,
                   threshold=0.9,
                   sigmoid_bound=False
                   ):
    """
    Evaluate model performance on a train/test split using custom region-based metrics.
    
    Trains a model on the training set and evaluates performance using custom metrics
    that measure the model's ability to predict high-yielding reaction regions. Generates
    predictions across all unique combinations of constant experimental variables in the
    test set, identifies predicted regions using threshold-based analysis, and compares
    with experimental optimum regions to compute region accuracy, precision, recall, and
    overlap metrics.
    
    Args:
        model : Model class
            Model instance with train(), predict(), and clear_model() methods
        id_var : str
            Column name for grouping variable (e.g., 'Substrate')
        dep_var : str
            Column name for dependent variable to predict (e.g., 'Yield')
        indep_var : str
            Column name for independent variable to vary predictions across (e.g., 'Temperature')
        constant_vars : list
            List of column names for constant experimental variables
        df_train : pd.DataFrame
            Training data DataFrame
        df_test : pd.DataFrame
            Test data DataFrame
        df_exp_optimum : pd.DataFrame
            Experimental optimum regions DataFrame with columns for each constant variable,
            id_var, and 'opt_Xmin', 'opt_Xmax' containing experimental region boundaries
        iter_step : float, optional
            Step size for interpolating predicted response curves (default: 0.1)
        threshold : float, optional
            Threshold (0-1) for identifying high-yielding predicted regions (default: 0.9)
        sigmoid_bound : bool, optional
            If True, applies yield transformations to bound predictions (default: False)
    
    Returns:
        pd.DataFrame
            DataFrame with one row per unique combination of constant variables and id_var
            containing columns:
            - All constant variables and id_var columns
            - 'mae', 'mse', 'r2score': Standard regression metrics
            - 'Xmin', 'Xmax': Experimental region boundaries
            - 'X_predmin', 'X_predmax': Predicted region boundaries
            - 'X_predopt': Independent variable value at predicted optimum
            - 'y_predopt': Predicted response value at optimum
            - 'accuracy': Fraction of experimental region correctly predicted
            - 'precision': Fraction of predicted region within experimental region
            - 'overlap': Overlap between predicted and experimental regions
            - 'recall': Fraction of experimental region correctly identified
            - 'midpoint_in_true_region': Boolean, whether predicted midpoint is in true region
            - 'max_in_true_region': Boolean, whether predicted optimum is in true region
            - 'X_pred': Interpolated independent variable values for predicted curve
            - 'y_pred': Predicted response values at interpolated points
    
    Notes:
        - Features are standardized using StandardScaler
        - One-hot encoded categorical variables are automatically detected
        - MLP models use interpolated predictions within indep_var range
        - Tree-based models use existing unique values or interpolation as fallback
        - Custom metrics are calculated using region_accuracy, region_precision, 
          region_recall, and region_overlap functions for each combination
    """
    #print(df_exp_optimum.columns)
    
    # Get all values the model will be trained on for the independent variable
    X_values_all = list(df_train[indep_var].values) + list(df_test[indep_var].values)
    X_values_unique = np.unique(X_values_all)
    #print(f"Unique values for {indep_var}: {X_values_unique}")

    # Initialize list to store metrics for each combination
    test_combination_metrics = []

    X_train, y_train = xy_split(df_train, id_var, dep_var)

    # Train the model
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model.clear_model()  # Clear the model to reset it

    
    if sigmoid_bound:
        y_train = yield_to_unbounded(y_train)

    model.train(X_train, y_train)

    if constant_vars is not type(list):
        list(constant_vars)

    if id_var not in constant_vars:
        constant_vars = [id_var] + constant_vars

    print(constant_vars)
    # Work out the unique combinations of constant variables from df_exp_optimum (non encoded)
    split_test_ids = df_test[id_var].unique()
    df_exp_optimum_test = df_exp_optimum[df_exp_optimum[id_var].isin(split_test_ids)]
    constant_var_values = {var: list(df_exp_optimum_test[var].unique()) for var in constant_vars}
    keys = list(constant_var_values.keys())
    values = list(constant_var_values.values())

    unique_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    for combination in unique_combinations:
        print(f"Evaluating combination: {combination}")
        
        # Check if the optimum region exists for combination in df_exp_optimum
        if df_exp_optimum.loc[df_exp_optimum[list(combination.keys())].eq(pd.Series(combination)).all(axis=1)].empty:
            print(f"No experimental optimum region found for this combination: {combination}. Skipping...")
            continue


        # Get the optimum region for the current combination of constant variables from df_exp_optimum
        combination_exp_row = df_exp_optimum.loc[df_exp_optimum[list(combination.keys())].eq(pd.Series(combination)).all(axis=1)]
        if combination_exp_row.empty:
            print(f"No experimental optimum region found for this combination: {combination}. Skipping...")
            continue

        Xmin = combination_exp_row['opt_Xmin'].values[0]
        Xmax = combination_exp_row['opt_Xmax'].values[0]


        df_test_subset = df_test.copy()
        
        for key, value in combination.items(): # Wittle down the df_test_subset to only the rows that match the current combination of constant variables
            if key not in df_test_subset.columns:
                
                ohe_col = f"{key}_{value}" # Assume the column was one-hot encoded
                df_test_subset = df_test_subset[df_test_subset[ohe_col] == 1]
            
            else:
                
                df_test_subset = df_test_subset[df_test_subset[key] == value]

    
        # Run predictions to generate the predicted response curve
        X_pred = df_test_subset[indep_var].values
        X_test_subset, y_test_subset = xy_split(df_test_subset, id_var, dep_var)
        
        #print(X_test_subset[:, 0])
        scaler_min = X_test_subset[:, 0].min()
        scaler_max = X_test_subset[:, 0].max()
        
        X_test_subset = scaler.transform(X_test_subset)
        
        y_pred = model.predict(X_test_subset)

        if sigmoid_bound:
            y_pred = unbounded_to_yield(y_pred)
        
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        # Standard metrics
        mae = mean_absolute_error(y_test_subset, y_pred)
        mse = mean_squared_error(y_test_subset, y_pred)
        r2score = r2_score(y_test_subset, y_pred)

        ind_min = df_test_subset[indep_var].min()
        ind_max = df_test_subset[indep_var].max()

        if str(model) == 'MultiLayerPerceptron':
            # Generate range for the independent variable
            
            granular_values = np.arange(ind_min, ind_max + iter_step, iter_step)
            
            X_pred_expanded, X_interpolated_pred = extend_X(df_test_subset, indep_var, id_var, dep_var, granular_values)
            y_interpolated_pred = model.predict(scaler.transform(X_pred_expanded)).ravel()
        else:
            granular_values = X_values_unique[(X_values_unique >= ind_min) & (X_values_unique <= ind_max)]

            X_pred_expanded, _ = extend_X(df_test_subset, indep_var, id_var, dep_var, granular_values = granular_values)    
            y_pred_curve = model.predict(scaler.transform(X_pred_expanded)).ravel()
            X_interpolated_pred, y_interpolated_pred = interpolate_data(granular_values, y_pred_curve, inter_step=iter_step)
        
        X_predmin, X_predmax = find_region(X_interpolated_pred, y_interpolated_pred, threshold=threshold)
        X_predopt, y_predopt = find_optimum(X_interpolated_pred, y_interpolated_pred)

        # Run custom metrics and append results
        accuracy, precision, overlap, recall, midpoint_in_true_region, max_in_true_region = run_custom_metrics(Xmin, Xmax, X_predmax, X_predmin, X_predopt, scaler_min=scaler_min, scaler_max=scaler_max)

        combination_metrics = {**combination,
                                 'mae': mae,
                                 'mse': mse,
                                 'r2score': r2score,
                                 'Xmin': Xmin,
                                 'Xmax': Xmax,
                                 'X_predmin': X_predmin,
                                 'X_predmax': X_predmax,
                                 'X_predopt': X_predopt,
                                 'y_predopt': y_predopt,
                                 'accuracy': accuracy,
                                 'precision': precision,
                                 'overlap': overlap,
                                 'recall': recall,
                                 'midpoint_in_true_region': midpoint_in_true_region,
                                 'max_in_true_region': max_in_true_region,
                                 'y_pred': y_interpolated_pred,
                                 'X_pred': X_interpolated_pred,}
        
        test_combination_metrics.append(combination_metrics)          


    return pd.DataFrame(test_combination_metrics)