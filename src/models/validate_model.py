import os
import pandas as pd
from src.utils.config import load_config
from src.utils.model_utils import get_validation_dfs, select_model_and_data, xy_split
from src.metrics.split_metrics import evaluate_split_standard, evaluate_split_custom


def main():
    """
    Main entry point for model validation with cross-validation and custom metrics.
    
    Performs cross-validation (leave-one-out or k-fold) on a selected model using
    a selected dataset. Evaluates model performance using standard metrics (MSE, MAE, R²)
    and optionally custom metrics based on experimental optimum regions. Saves detailed
    validation results and updates a validation log file.
    
    Returns:
        None (saves validation metrics to CSV files and updates validation log)
    """
    # Load configuration files
    config_files = ["configs/base.yaml", "configs/models/validate_model.yaml"]
    cfg = load_config(config_files)

    save_metric = cfg['metrics']['save']
    update_log = cfg['metrics']['update_log']
    metrics_dir = cfg['output']['metrics_dir']
    val_method = cfg['validation']['val_method']

    input_dir = cfg['data']['model_input_dir']

    # Select model and data, then load and initialise
    model, df, model_name, dset_name  = select_model_and_data(input_dir)

    

    x_values, y_values = xy_split(df, cfg['validation']['id_col'], cfg['validation']['dep_vars'])

    n_features = x_values.shape[1]

    # If model is a neural network, set input shape and output units based on the data
    if hasattr(model, 'input_shape') and hasattr(model, 'output_units'):
        input_len = x_values.shape[1]  # Exclude constant variables from input shape
        model.input_shape = (input_len, )
        model.output_units = y_values.shape[1]
    
    # Generate splitter base on val_method in config file
    if val_method == 'leave_one_out':
        from sklearn.model_selection import LeaveOneOut
        splitter = LeaveOneOut()
        print("Using Leave-One-Out cross-validation for validation (change in config file validate_model.yaml).")
    elif val_method == 'k_fold':
        from sklearn.model_selection import KFold
        shuffle = cfg['validation'].get('shuffle', True)
        random_state = cfg['validation'].get('random_state', 42)
        splitter = KFold(n_splits=cfg['validation']['cv_folds'], shuffle=shuffle, random_state=random_state)
        print(f"Using K-Fold cross-validation with {cfg['validation']['cv_folds']} folds for validation (change in config file validate_model.yaml).")
    else:
        raise ValueError(f"Validation method {val_method} not supported. Use 'leave_one_out' or 'k_fold'.")


    # Generate train/test data splits to itterate through
    train_dfs, test_dfs = get_validation_dfs(df, cfg['validation']['id_col'], splitter)


    if cfg['validation']['custom']:
        opt_path = cfg['data']['exp_optimum_csv_path']
        df_exp_optimum = pd.read_csv(opt_path)
        print(f"Using custom metrics for validation with experimental optimum regions from {opt_path}.")


    # Itterate through train/test splits and calculate metrics for each experiment. Append dataframes to a list...
    split_metrics_dfs = []
    for i, (df_train, df_test) in enumerate(zip(train_dfs, test_dfs)):

        split_metrics = evaluate_split_custom(
            model,
            cfg['validation']['id_col'],
            cfg['validation']['dep_vars'],
            cfg['validation']['indep_vars'],
            cfg['validation']['constant_vars'],
            df_train,
            df_test,
            df_exp_optimum if cfg['validation']['custom'] else None,
            iter_step=cfg['metrics']['iter_step'],
            threshold=cfg['metrics']['threshold'],
            sigmoid_bound=cfg['validation']['sigmoid_bound'] if 'sigmoid_bound' in cfg['metrics'] else False
        )
        # Add info about the split
        #print(type(split_metrics))
        split_metrics['fold'] = i + 1
        split_metrics_dfs.append(split_metrics)
        #print(len(split_metrics.keys()))
        #print(len(split_metrics_dfs))
    
    # Combine the info from each test/train split into a single DataFrame
    metric_df = pd.concat(split_metrics_dfs, ignore_index=True, axis = 0)


    model_metrics = {
    'mean_mse': metric_df['mse'].mean() if 'mse' in metric_df.columns else None,
    'mean_mae': metric_df['mae'].mean() if 'mae' in metric_df.columns else None,
    'mean_r2': metric_df['r2score'].mean() if 'r2score' in metric_df.columns else None,
    'mean_accuracy': metric_df['accuracy'].mean() if 'accuracy' in metric_df.columns else None,
    'mean_precision': metric_df['precision'].mean() if 'precision' in metric_df.columns else None,
    'mean_overlap': metric_df['overlap'].mean() if 'overlap' in metric_df.columns else None,
    'mean_recall': metric_df['recall'].mean() if 'recall' in metric_df.columns else None,
    'midpoint_hit_rate': metric_df['midpoint_in_true_region'].sum() / metric_df['midpoint_in_true_region'].count() if 'midpoint_in_true_region' in metric_df.columns else None,
    'max_hit_rate': metric_df['max_in_true_region'].sum() / metric_df['max_in_true_region'].count() if 'max_in_true_region' in metric_df.columns else None,
    }
    print(f"Metrics for {model_name} trained with {dset_name} and {val_method} validation...")
    print(f"{model_metrics}")

    timestamp = pd.Timestamp.now().strftime('%d-%m-%Y_%H.%M')
    
    # Save the metrics (granular info) DataFrame to a CSV file if configured
    if save_metric:
        if val_method == 'leave_one_out':
            metric_df_filename = f"{model_name}_{dset_name}_loo_{timestamp}.csv"
        elif val_method == 'k_fold':
            metric_df_filename = f"{model_name}_{dset_name}_kfold{cfg['validation']['cv_folds']}_{timestamp}.csv"
        
        metric_df_path = os.path.join(metrics_dir, metric_df_filename)
        #print(metric_df_path)
        #print(metric_df.head())
        metric_df.to_csv(metric_df_path, index=False)

    # Update the log CSV file with the model metrics if configured. Comparison between using different models and datasets
    if update_log:
        from src.utils.log_utils import update_log_csv
        log_path = cfg['output']['val_log_path']
        new_entry = {'model': model_name,
                     'model_params': model.model_params if hasattr(model, 'model_params') else {},
                       'dataset': dset_name,
                       'n_features': n_features,
                       'val_method': val_method,
                       'timestamp': timestamp,}
        new_entry.update(model_metrics)
        update_log_csv(log_path, new_entry)


if __name__ == "__main__":
    main()