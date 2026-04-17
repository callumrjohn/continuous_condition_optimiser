import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.preprocessing.melt import melt_data_df
from src.metrics.curve_analysis import interpolate_data, find_region, find_optimum


def main():
    """
    Extract experimental optimum regions from raw experimental yield data.
    
    Orchestrates the identification of optimal experimental conditions by analyzing
    yield curves for each unique combination of experimental variables. Loads configuration
    from YAML files, melts experimental data into long format if needed, identifies
    high-performing regions using threshold-based curve analysis, and saves results
    to CSV for use in model evaluation metrics.
    
    Configuration keys (from configs/metrics/get_optimums.yaml):
        - data.yield_path: Path to input experimental data CSV
        - metrics.output_path: Path to save optimum regions CSV
        - metrics.threshold: Yield threshold (0-1) for defining high-performing region
        - metrics.iter_step: Step size for interpolating yield curves
        - metrics.success_cutoff: Minimum yield value to process combination
        - preprocessing.id_vars: List of columns defining experiment combinations
        - preprocessing.melt: Whether to reshape data from wide to long format
        - preprocessing.var_name: Column name for varying experimental conditions
        - preprocessing.value_name: Column name for yield response values
        - preprocessing.drop_nan: Whether to remove NaN values before analysis
    
    Returns:
        None
        Saves results to CSV with columns:
        - All id_vars columns (e.g., Substrate, Reagent, etc.)
        - 'opt_Xmin': Minimum boundary of high-yielding region
        - 'opt_Xmax': Maximum boundary of high-yielding region
        - 'opt_X': Independent variable value at experimental optimum
        - 'opt_y': Experimental yield value at optimum
    
    Notes:
        - Combinations where all experiments yield < success_cutoff are skipped
        - Interpolation uses linear interpolation at iter_step intervals
        - High-performing region is defined as all points >= threshold * max(yield)
        - Used as reference for comparing model predictions against true optima
        - This data is required for the custom metrics evaluation in evaluate_split_custom()
    """
    
    config_files = ["configs/base.yaml", "configs/metrics/get_optimums.yaml"]
    cfg = load_config(config_files)

    input_path = cfg['data']['yield_path']

    output_path = cfg['metrics']['output_path']
    threshold = cfg['metrics']['threshold']
    iter_steps = cfg['metrics']['iter_step']
    success_cutoff = cfg['metrics']['success_cutoff']

    id_vars = cfg['preprocessing']['id_vars']
    melt = cfg['preprocessing']['melt']
    var_name = cfg['preprocessing']['var_name']
    value_name = cfg['preprocessing']['value_name']
    drop_nan = cfg['preprocessing']['drop_nan']



    if melt:
        df = melt_data_df(pd.read_csv(input_path), id_vars=id_vars, var_name=var_name, value_name=value_name, drop_nan=drop_nan)
    else:
        df = pd.read_csv(input_path)

    unique_combinations = list(df[id_vars].drop_duplicates().itertuples(index=False, name=None))
    #print(unique_combinations)
    
    results = []
    for combination in unique_combinations:
        condition = {id_vars[i]: combination[i] for i in range(len(id_vars))}
        mask = np.logical_and.reduce([df[k] == v for k, v in condition.items()])
        #print(combination)
        df_subset = df[mask]
        #print(df_subset)
        if drop_nan:
            #print('Drop NaN')
            df_subset = df_subset.dropna(subset=[value_name])

        X = df_subset[var_name].values
        #print(X)
        y = df_subset[value_name].values

        if success_cutoff > max(y):
            print(f"Skipping {combination} due to {value_name} being less than {success_cutoff} for all experiments.")
            continue

        # Generate the interpolated data and find the optimum region
        X_interpolated, y_interpolated = interpolate_data(X, y, inter_step=iter_steps)
        opt_Xmin, opt_Xmax = find_region(X_interpolated, y_interpolated, threshold=threshold)
        opt_X, opt_y = find_optimum(X, y)

        # Append the optimum region, along with the best experiment, for this combination
        id_vars_dict = dict(zip(id_vars, combination))
        id_vars_dict.update({
            'opt_Xmin': opt_Xmin,
            'opt_Xmax': opt_Xmax,
            'opt_X': opt_X,
            'opt_y': opt_y
        })
        results.append({
            **id_vars_dict
        })


    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()