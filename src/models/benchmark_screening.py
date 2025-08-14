import pandas as pd
import numpy as np
from src.utils.config import load_config
from src.utils.model_utils import get_validation_dfs, select_model_and_data, xy_split
from sklearn.model_selection import KFold
from src.metrics.split_metrics import evaluate_split_standard

def main():

    config_files = ["configs/base.yaml", "configs/models/benchmark_screening.yaml"]
    cfg = load_config(config_files)

    input_dir = cfg['data']['model_input_dir']

    save_metric = cfg['metrics']['save']
    id_col = cfg['validation']['id_col']
    dep_vars = cfg['validation']['dep_vars']
    #indep_vars = cfg['validation']['indep_vars']
    #constant_vars = cfg['validation']['constant_vars']
    
    cv_folds = cfg['validation']['cv_folds']
    shuffle = cfg['validation'].get('shuffle', True)
    random_state = cfg['validation'].get('random_state', 42)

    # Select model and data, then load and initialise
    model, df, model_name, dset_name  = select_model_and_data(input_dir)

    kf = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)

    #train_dfs, test_dfs = kf.split(df)

    maes, mses, r2scores = [], [], []
    for train_idx, test_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        split_metric = evaluate_split_standard(model,
                                                  id_col,
                                                  dep_vars,
                                                  train_df,
                                                  test_df,
                                                  sigmoid_bound=False)

        maes.append(split_metric['mae'])
        mses.append(split_metric['mse'])
        r2scores.append(split_metric['r2score'])
    
    mae = np.mean(maes)
    mse = np.mean(mses)
    r2score = np.mean(r2scores)
    model_metrics = {
        'mae': mae,
        'mse': mse,
        'r2score': r2score
    }

    # Save metrics
    if save_metric:
        from src.utils.log_utils import update_log_csv
        
        timestamp = pd.Timestamp.now().strftime('%d-%m-%Y_%H.%M')

        log_path = cfg['output']['val_log_path']
        new_entry = {'model': model_name,
                        'model_params': model.model_params if hasattr(model, 'model_params') else {},
                        'dataset': dset_name,
                        'timestamp': timestamp,}
        new_entry.update(model_metrics)
        update_log_csv(log_path, new_entry)


if __name__ == "__main__":
    main()
