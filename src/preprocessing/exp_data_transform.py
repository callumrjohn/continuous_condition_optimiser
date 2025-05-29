import pandas as pd
from src.utils.config import load_config
from src.preprocessing.melt import melt_data_df
from src.preprocessing.encode import one_hot_encode


def main():

    # Load configs
    config_files = ["configs/base.yaml", "configs/preprocessing/exp_data_transform.yaml"]
    cfg = load_config(config_files)

    # Input and output paths
    input_path = cfg['data']['yield_path']
    output_path = cfg['preprocessing']['output_path']
    
    # Melt parameters
    id_vars = cfg['preprocessing']['id_vars']
    var_name = cfg['preprocessing']['var_name']
    value_name = cfg['preprocessing']['value_name']
    drop_nan = cfg['preprocessing']['drop_nan']

    # One-hot encoding parameters
    ohe_columns = cfg['preprocessing']['one_hot_encode_columns']
    drop_first = cfg['preprocessing']['drop_first']
    sparse_encoded = cfg['preprocessing']['sparse_encode']

    # What functions to apply
    melt = cfg['preprocessing']['melt']
    encode = cfg['preprocessing']['encode']
    
    df = pd.read_csv(input_path)

    if melt:
        df = melt_data_df(df, id_vars=id_vars, var_name=var_name, value_name=value_name, drop_nan=drop_nan)
    if encode:
        df = one_hot_encode(df, columns=ohe_columns, drop_first=drop_first, sparse_encoded=sparse_encoded)

    if not (melt or encode):
        raise ValueError("Either 'melt' or 'encode' must be True in the configuration (configs/base.yaml).")

    #reorder to have value name as the last column
    cols = [col for col in df.columns if col != value_name] + [value_name]
    df = df[cols]
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()