import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestRegressor
from src.utils.config import load_config
from src.utils.model_utils import select_input_data
from sklearn.utils import resample

def pca_reduce_dataframe(df, id_col, target_cols, n_components):
    """
    Reduce DataFrame dimensionality using Principal Component Analysis (PCA).
    
    Applies PCA to compress features while preserving maximum variance. Returns
    DataFrame with principal components plus ID and target columns.
    
    Args:
        df: Input DataFrame
        id_col: Column name for unique identifiers
        target_cols: List of target variable column names
        n_components: Number of principal components to keep
    
    Returns:
        DataFrame with PCA-reduced features plus id_col and target columns.
        Columns ordered as: [id_col, target_cols, PC1, PC2, ...]
    """
    # Separate features, id, and targets
    feature_cols = [col for col in df.columns if col not in ([id_col] + target_cols)]
    X = df[feature_cols].values
    ids = df[id_col].values
    targets = df[target_cols].values

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Create new DataFrame with reduced features
    pca_cols = [f'PC{i+1}' for i in range(n_components)]
    reduced_df = pd.DataFrame(X_pca, columns=pca_cols)
    reduced_df[id_col] = ids
    for i, col in enumerate(target_cols):
        reduced_df[col] = targets[:, i] if targets.ndim > 1 else targets

    # Reorder columns: id, targets, PCs
    cols_order = [id_col] + target_cols + pca_cols
    reduced_df = reduced_df[cols_order]
    return reduced_df



def rfe_reduce_dataframe(df, id_col, target_cols, n_features_to_select, cv=5, step=10):
    """
    Reduce DataFrame features using Recursive Feature Elimination (RFE).
    
    Selects the most important features for the target variable using RFE with
    a RandomForest estimator. Can use cross-validation to automatically determine
    optimal number of features (if n_features_to_select='auto').
    
    Args:
        df: Input DataFrame
        id_col: Column name for unique identifiers
        target_cols: List of target variable column names (typically single element)
        n_features_to_select: Number of features to select. Can be:
            - Integer: exact number of features to select
            - 'auto': use RFECV to find optimal number using cross-validation
        cv: Number of cross-validation folds for RFECV (default: 5)
        step: Number of features to remove at each iteration (default: 10)
    
    Returns:
        DataFrame with selected features plus id_col and target columns.
        Columns ordered as: [id_col, target_cols, selected_features]
    
    Raises:
        ValueError: If n_features_to_select is not 'auto' or an integer
    """
    # Separate features, id, and targets
    feature_cols = [col for col in df.columns if col not in ([id_col] + target_cols)]
    X = df[feature_cols].values
    ids = df[id_col].values
    targets = df[target_cols].values

    # Use a simple estimator for RFE (can be changed)
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    if n_features_to_select == 'auto':
        rfe = RFECV(estimator, step=step, cv=cv, scoring='neg_mean_squared_error')
    elif isinstance(n_features_to_select, int):
        rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
    else:
        raise ValueError("n_features_to_select must be 'auto' or an integer.")
    
    X_rfe = rfe.fit_transform(X, targets.ravel() if targets.ndim > 1 and targets.shape[1] == 1 else targets)

    # Get selected feature names
    selected_features = [feature for feature, selected in zip(feature_cols, rfe.support_) if selected]

    # Create new DataFrame with selected features
    reduced_df = pd.DataFrame(X_rfe, columns=selected_features)
    reduced_df[id_col] = ids
    for i, col in enumerate(target_cols):
        reduced_df[col] = targets[:, i] if targets.ndim > 1 else targets

    # Reorder columns: id, targets, selected features
    cols_order = [id_col] + target_cols + selected_features
    reduced_df = reduced_df[cols_order]
    return reduced_df


def rfe_reduce_dataframe_bootstrap(
    df, id_col, target_cols, n_features_to_select, 
    n_bootstraps=10, step=10, random_state=42
):
    """
    Reduce DataFrame features using bootstrapped Recursive Feature Elimination (RFE).
    
    Performs RFE multiple times on bootstrap samples of the data and selects features
    based on average ranking across all bootstrap iterations.
    
    Args:
        df: Input DataFrame
        id_col: Column name for unique identifiers
        target_cols: List of target variable column names (typically single element)
        n_features_to_select: Number of features to select
        n_bootstraps: Number of bootstrap samples to generate (default: 10)
        step: Number of features to remove at each RFE iteration (default: 10)
        random_state: Random seed for reproducible bootstrap sampling (default: 42)
    
    Returns:
        DataFrame with selected features plus id_col and target columns.
        Columns ordered as: [id_col, target_cols, selected_features]
    
    Notes:
        - Each bootstrap iteration uses a different random_state (random_state + i)
        - Features are ranked by average rank across all bootstraps
    """
    # Separate features, id, and targets
    feature_cols = [col for col in df.columns if col not in ([id_col] + target_cols)]
    X = df[feature_cols].values
    ids = df[id_col].values
    targets = df[target_cols].values
    y = targets.ravel() if targets.ndim > 1 and targets.shape[1] == 1 else targets

    # Set up rank accumulation
    n_features = X.shape[1]
    rank_accumulator = np.zeros(n_features)

    # Perform bootstrapping
    for i in range(n_bootstraps):
        # Bootstrap resample
        X_resampled, y_resampled = resample(X, y, random_state=random_state + i)

        # Fit RFE with RF on bootstrap sample
        estimator = RandomForestRegressor(n_estimators=100, random_state=random_state + i)
        rfe = RFE(estimator, n_features_to_select=min(n_features_to_select, n_features), step=step)
        rfe.fit(X_resampled, y_resampled)

        # Accumulate feature ranks
        rank_accumulator += rfe.ranking_

    # Average ranks and select top features
    avg_ranks = rank_accumulator / n_bootstraps
    selected_indices = np.argsort(avg_ranks)[:n_features_to_select]
    selected_features = [feature_cols[i] for i in selected_indices]

    # Create new DataFrame with selected features
    reduced_df = pd.DataFrame(df[selected_features].values, columns=selected_features)
    reduced_df[id_col] = ids
    for i, col in enumerate(target_cols):
        reduced_df[col] = targets[:, i] if targets.ndim > 1 else targets

    # Reorder columns: id, targets, selected features
    cols_order = [id_col] + target_cols + selected_features
    reduced_df = reduced_df[cols_order]
    return reduced_df


def main():
    """
    Main entry point for feature reduction preprocessing workflow.
    
    Loads configuration, prompts user to select input data, and applies either
    RFE or bootstrapped RFE for feature selection. Saves the reduced dataset
    with only selected features retained.
    
    Returns:
        None (saves reduced DataFrame to CSV file)
    """
    config_files = ["configs/base.yaml", "configs/preprocessing/feature_reduction.yaml"]
    cfg = load_config(config_files)

    input_data_folder = cfg['data']['model_input_dir']
    id_col = cfg['preprocessing']['join_key']

    value_name = cfg['preprocessing']['value_name']

    feature_reduction_method = cfg['preprocessing']['feature_reduction_method']
    features_to_select = cfg['preprocessing']['feature_reduction']['rfe']['n_features_to_select']
    step = cfg['preprocessing']['feature_reduction']['rfe']['step']
    cv_folds = cfg['preprocessing']['feature_reduction']['rfe']['cv_folds']

    df, input_data_name = select_input_data()

    if feature_reduction_method == 'rfe':

        reduced_df = rfe_reduce_dataframe(
            df, 
            id_col=id_col, 
            target_cols=[value_name], 
            n_features_to_select=features_to_select, 
            step=step,
            cv=cv_folds
        )
    elif feature_reduction_method == 'rfe_bootstrap':

        n_bootstraps = cfg['preprocessing']['feature_reduction']['rfe_bootstrap']['n_bootstraps']
        reduced_df = rfe_reduce_dataframe_bootstrap(
            df, 
            id_col=id_col, 
            target_cols=[value_name], 
            n_features_to_select=features_to_select, 
            
            n_bootstraps=n_bootstraps, 
            step=step
        )
    else:
        raise ValueError(f"Feature reduction method '{feature_reduction_method}' not supported. Use 'rfe'.")

    df_features = df.shape[1] - 2  # Exclude id and target columns
    reduced_df_features = reduced_df.shape[1] - 2  # Exclude id and target columns
    reduced_df_name = f"{input_data_name}_{feature_reduction_method}_{str(features_to_select)}.csv"

    output_path = f"{input_data_folder}/{reduced_df_name}"

    reduced_df.to_csv(output_path, index=False)
    print(f"Reduced DataFrame saved to {output_path}")
    print(f"Original feature count: {df_features}, Reduced feature count: {reduced_df_features}")

if __name__ == "__main__":
    main()
