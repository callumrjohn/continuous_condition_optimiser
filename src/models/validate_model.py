import os
import sys
import itertools
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.config import load_config
from src.utils.model_utils import load_model_class, xy_split, select_model_and_data
from src.guis.data_model_selector import select_data_models_tkinter
from src.models.train_model import xy_split
from src.metrics.curve_analysis import interpolate_data, find_region, find_optimum
from src.metrics.custom_scoring import run_custom_metrics



# Split data into training and testing sets using standard strategy - Split by unique ID (eg Substrate) only
def train_test_split_id(df, id_col, test_size = 0.2, random_state = 42):

    unique_ids = df[id_col].unique()

    np.random.seed(random_state)
    
    np.random.shuffle(unique_ids)
    split_index = int(len(unique_ids) * (1 - test_size))

    train_ids = unique_ids[:split_index]
    test_ids = unique_ids[split_index:]

    train_df = df[df[id_col].isin(train_ids)]
    test_df = df[df[id_col].isin(test_ids)]

    return train_df, test_df


# Split data into training and testing sets using Leave-One-Out (LOO) strategy - Split by unique ID (eg Substrate) only
def loo_split(id, df, id_col):

    train_df = df[df[id_col] != id]
    test_df = df[df[id_col] == id]

    return train_df, test_df






def tts_validation_stf(model, 
                   df, 
                   id, 
                   target_columns, 
                   test_size = 0.2, 
                   cv_folds = 5, 
                   random_state = 42
                   ):
    
    maes = []
    mses = []
    r2scores = []

    for _ in range(cv_folds):
        
        # Split the data into training and testing sets
        train_df, test_df = train_test_split_id(df, id, target_columns, test_size, random_state)
        X_train, y_train = xy_split(train_df, id, target_columns)
        X_test, y_test = xy_split(test_df, id, target_columns)

        # Train the model
        model.train(X_train, y_train)
        
        # Calculate standard metrics
        y_pred = model.predict(X_test)
        maes.append(mean_absolute_error(y_test, y_pred))
        mses.append(mean_squared_error(y_test, y_pred))
        r2scores.append(r2_score(y_test, y_pred))
    
    # Calculate the average of the metrics
    avg_mae = np.mean(maes)
    avg_mse = np.mean(mses)
    avg_r2 = np.mean(r2scores)

    print(f"Average MAE: {avg_mae}, Average MSE: {avg_mse}, Average R2: {avg_r2} over {cv_folds} folds")
    
    return {'mae': avg_mae, 'mse': avg_mse, 'r2': avg_r2}



def tts_validation_custom(model, 
                   df, 
                   id_var,
                   contant_vars,
                   dep_var,
                   target_column, 
                   cv_folds = 5,
                   iter_steps = 0.1,
                   threshold = 0.9
                   ):
    
    # Load the experimental optimum region CSV file
    config_files = ["configs/base.yaml"]
    cfg = load_config(config_files)
    exp_optimum_csv_path = cfg['data']['exp_optimum_csv_path']
    if not os.path.exists(exp_optimum_csv_path):
        raise FileNotFoundError(f"Experimental optimum region CSV file not found at {exp_optimum_csv_path}. Generate using get_optimums.py first.")
    df_exp_optimum = pd.read_csv(exp_optimum_csv_path)



    # Initialize lists to store metrics
    accuracies = []
    precisions = []
    overlaps = []
    recalls = []
    midpoint_in_true_regions = []
    max_in_true_regions = []


    # Perform cross-validation
    for _ in range(cv_folds):

        unique_ids = df[id_var].drop_duplicates().values
        
        for id in unique_ids:
            
            df_train, df_test = loo_split(id, df, id_var)

            X_train, y_train = xy_split(df_train, id_var, target_column)

            # Train the model
            model.train(X_train, y_train)

            constant_var_values = {var: list(df_exp_optimum[var].unique()) for var in contant_vars}
            keys = list(constant_var_values.keys())
            values = list(constant_var_values.values())
            unique_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

            
            for combination in unique_combinations:
                
                
                df_test_subset = df_test.copy()
                
                for key, value in combination.items(): # Wittle down the df_test_subset to only the rows that match the current combination of constant variables
                    if key not in df_test_subset.columns:
                        
                        ohe_col = f"{key}_{value}" # Assume the column was one-hot encoded
                        df_test_subset = df_test_subset[df_test_subset[ohe_col] == 1]
                    
                    else:
                        
                        df_test_subset = df_test_subset[df_test_subset[key] == value]

            

                X_pred = df_test_subset[dep_var].values
                X_train_subset, _ = xy_split(df_test_subset, id_var, target_column)
                y_pred = model.predict(X_train_subset)

                X_interpolated_pred, y_interpolated_pred = interpolate_data(X_pred, y_pred, inter_step=iter_steps)
                opt_Xmin_pred, opt_Xmax_pred = find_region(X_interpolated_pred, y_interpolated_pred, threshold=threshold)
                opt_X_pred, opt_y_pred = find_optimum(X_pred, y_pred)

                accuracy, precision, overlap, recall, midpoint_in_true_region, max_in_true_region = run_custom_metrics(X





                    



            # Calculate custom metrics for each sample in the test set
                
                # Get experimental optium region information
                opt_Xmin, optXmax = df_exp_optimum.loc[
                    (df_exp_optimum[id_var] == pd.Series(combination)).all(axis=1),
                    ['opt_Xmin', 'opt_Xmax']
                ].values[0]
                
                df_train, df_test = loo_split(combination[0], df, id_var)
                
                mask = np.logical_and.reduce([df[k] == v for k, v in condition.items()])
                df_test_subset = df[mask]

                if df_test_subset.empty:
                    continue

       








def main():

    model, df, model_name, training_dataset, cfg = select_model_and_data()
   
    id_col = cfg['training']['columns_to_remove']
    target_columns = cfg['training']['target_variables']

    

    # Seperate the model into features and target variables and train...
    X, y = xy_split(df, remove_columns, target_columns)
    model.train(X, y)

    # Save the model for later use
    model_save_path = cfg['output']['output_model_dir'] + f"{model_name.lower()}_{training_dataset}.pkl"
    print(model_save_path)
    model.save(model_save_path)

    print(f"Model {model_name} trained and saved to {model_save_path}")






