import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from src.utils.data_utils import yield_to_unbounded, unbounded_to_yield
from src.utils.model_utils import xy_split, extend_X
from src.metrics.curve_analysis import interpolate_data, find_region, find_optimum
from src.metrics.custom_metrics import region_accuracy, region_precision, region_overlap, region_recall, is_midpoint_in_true_region, is_max_in_true_region


# RUN THEM ALL TOGETHER
def run_custom_metrics(Xmin, Xmax, X_predmax, X_predmin, X_predopt, scaler_min=0, scaler_max=25):
    """
    Calculate custom metrics for the predicted high yielding region.
    """
    accuracy = region_accuracy(Xmin, Xmax, X_predmax, X_predmin, scaler_min=scaler_min, scaler_max=scaler_max)
    precision = region_precision(Xmin, Xmax, X_predmax, X_predmin, scaler_min=scaler_min, scaler_max=scaler_max)
    overlap = region_overlap(Xmin, Xmax, X_predmin, X_predmax)
    recall = region_recall(Xmin, Xmax, X_predmin, X_predmax)
    midpoint_in_true_region = is_midpoint_in_true_region(Xmin, Xmax, X_predmin, X_predmax)
    max_in_true_region = is_max_in_true_region(Xmin, Xmax, X_predopt)

    return accuracy, precision, overlap, recall, midpoint_in_true_region, max_in_true_region

# Calcuate metrics for a test/train slit using standard metrics
def evaluate_split_standard(model,
                   id_var,
                   dep_vars,
                   train_df,
                   test_df,
                   sigmoid_bound=False):
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


# Calculate metrics for a test/train slit using custom metrics. Return as a DataFrame
def evaluate_split_custom(model, # Model to evaluate (class)
                   id_var, # ID variable to split on (eg Substrate)
                   dep_var, # Dependent variable to predict (eg Yield, conversion) - there can only be one...
                   indep_var, # Independent variable to preduct (eg Temperature, pH, acid equivalents, etc) - there can only be one...
                   constant_vars, # Constant variables used in the experiment
                   df_train, # DataFrame of training data
                   df_test, # DataFrame of test data
                   df_exp_optimum, # DataFrame of experimental optimum regions (non-encoded) - Generated using get_optimums.py
                   iter_step=0.1,
                   threshold=0.9,
                   sigmoid_bound=False
                   ):
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