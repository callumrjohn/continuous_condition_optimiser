import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.preprocessing.melt import melt_data_df
from src.metrics.curve_analysis import interpolate_data, find_region, find_optimum


def main():
    
    config_files = ["configs/base.yaml", "configs/metrics/get_optimums.yaml"]
    cfg = load_config(config_files)

    input_path = cfg['data']['yield_path']

    output_path = cfg['metrics']['output_path']
    threshold = cfg['metrics']['threshold']
    iter_steps = cfg['metrics']['iter_step']

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