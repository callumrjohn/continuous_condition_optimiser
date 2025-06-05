import numpy as np
import pandas as pd


#----------------- Test function for melting data with a continuous variable ----------------
def test_melt_data_df_with_cont_var():
    from src.preprocessing.melt import melt_data_df

    # Create a mock DataFrame
    df = pd.DataFrame({
        'id': ['substrate1', 'substrate2', 'substrate3'],
        '1': [10, 20, 30],
        '2': [20, 40, np.nan]
    })

    # Melt the DataFrame
    melted_df = melt_data_df(df, id_vars=['id'], var_name='acid_equivalents', value_name='yield')

    # Check the shape of the melted DataFrame
    assert melted_df.shape == (5, 3)  # 3 rows * 2 features = 6 rows

    # Check the columns
    assert list(melted_df.columns) == ['id', 'acid_equivalents', 'yield']

    # Check the values
    expected_values = [
        ('substrate1', 1.0, 10.0),
        ('substrate2', 1.0, 20.0),
        ('substrate3', 1.0, 30.0),
        ('substrate1', 2.0, 20.0),
        ('substrate2', 2.0, 40.0)
    ]
    
    for i, (id_val, feature_val, value_val) in enumerate(expected_values):
        assert melted_df.iloc[i].tolist() == [id_val, feature_val, value_val]



#----------------- Test function for melting data with a categorical variable ----------------
def test_melt_data_df_with_cat_var():
    from src.preprocessing.melt import melt_data_df

    # Create a mock DataFrame
    df = pd.DataFrame({
        'id': ['substrate1', 'substrate2', 'substrate3'],
        'cat1': [10, np.nan, 60],
        'cat2': [10, 20, 30]
    })

    # Melt the DataFrame
    melted_df = melt_data_df(df, id_vars=['id'], var_name='catalyst', value_name='yield', drop_nan=True)

    # Check the shape of the melted DataFrame
    assert melted_df.shape == (5, 3)

    # Check the columns
    assert list(melted_df.columns) == ['id', 'catalyst', 'yield']

    # Check the values
    expected_values = [
        ('substrate1', 'cat1', 10.0),
        ('substrate3', 'cat1', 60.0),
        ('substrate1', 'cat2', 10.0),
        ('substrate2', 'cat2', 20.0),
        ('substrate3', 'cat2', 30.0)
    ]
    
    for i, (id_val, feature_val, value_val) in enumerate(expected_values):
        assert melted_df.iloc[i].tolist() == [id_val, feature_val, value_val]


#----------------- Test function for one-hot encoding ----------------
def test_one_hot_encode():
    from src.preprocessing.encode import one_hot_encode

    # Create a mock DataFrame
    df = pd.DataFrame({
        'id': ['substrate1', 'substrate2', 'substrate3', 'substrate3'],
        'category': ['cat1', 'cat2', 'cat1', 'cat3'],
        'variable': [1, 2, 3, 4]
    })

    # One-hot encode the 'category' column
    encoded_df = one_hot_encode(df, columns=['category'], drop_first = False, sparse_encoded = False)

    # Check the shape of the encoded DataFrame
    assert encoded_df.shape == (4, 5)

    # Check the columns
    assert list(encoded_df.columns) == ['id', 'variable', 'category_cat1', 'category_cat2', 'category_cat3']

    # Check the values
    expected_values = [
        ('substrate1', 1, 1, 0, 0),
        ('substrate2', 2, 0, 1, 0),
        ('substrate3', 3, 1, 0, 0),
        ('substrate3', 4, 0, 0, 1)
    ]
    
    for i, (id_val, cat1_val, cat2_val, cat3_val, var_value) in enumerate(expected_values):
        assert encoded_df.iloc[i].tolist() == [id_val, cat1_val, cat2_val, cat3_val, var_value]



#----------------- Test function for merging data with duplicate_selection = 'first'----------------
def test_merge_data_first():
    from src.preprocessing.merge_data import merge_dfs


    # Create mock DataFrames
    data = pd.DataFrame({
        'id': ['substrate1', 'substrate2', 'substrate3'],
        'reagent_one': [1, 0, 0],
        'reagent_two': [0, 1, 0],
        'reagent_three': [0, 0, 1],
        'yield': [10, 20, 30]
    })

    fingerprints = [
        pd.DataFrame({
            'id': ['substrate1', 'substrate2', 'substrate3'],
            'fp1': [0.1, 0.2, 0.3],
            'fp2': [0.4, 0.5, 0.6],
            'fp3': ['no', 'yes', 'yes']
        }),
        pd.DataFrame({
            'id': ['substrate1', 'substrate2', 'substrate3'],
            'fp2': [0.4, 0.5, 0.8],
            'fp3': ['no', 'yes', 'Yes'],
            'fp4': [0.7, 0.8, 0.9],
            'fp5': [1.0, 1.1, 1.2]
        })
    ]

    # Merge the DataFrames
    merged_df = merge_dfs(data, fingerprints, id_col='id', how='inner', desc_labels=None, duplicate_selection='first')

    # Check the shape of the merged DataFrame
    assert merged_df.shape == (3, 10)

    # Check the columns
    assert list(merged_df.columns) == ['id', 'reagent_one', 'reagent_two', 'reagent_three', 'yield', 'fp1_0', 'fp2_0', 'fp3_0', 'fp4_1', 'fp5_1']

    # Check the values
    expected_values = [
        ('substrate1', 1, 0, 0, 10, 0.1, 0.4, 'no', 0.7, 1.0),
        ('substrate2', 0, 1, 0, 20, 0.2, 0.5, 'yes', 0.8, 1.1),
        ('substrate3', 0, 0, 1, 30, 0.3, 0.6, 'yes', 0.9, 1.2)
    ]


    for i, (id_val, r_1, r_2, r_3, y, fp1_val, fp2_val, fp3_val, fp4_val, fp5_val) in enumerate(expected_values):
        assert merged_df.iloc[i].tolist() == [id_val, r_1, r_2, r_3, y, fp1_val, fp2_val, fp3_val, fp4_val, fp5_val]



#----------------- Test function for merging data with duplicate_selection = 'last'----------------
def test_merge_data_last():
    from src.preprocessing.merge_data import merge_dfs


    # Create mock DataFrames
    data = pd.DataFrame({
        'id': ['substrate1', 'substrate2', 'substrate3'],
        'yield': [10, 20, 30]
    })

    fingerprints = [
        pd.DataFrame({
            'id': ['substrate1', 'substrate2', 'substrate3'],
            'fp1': [0.1, 0.2, 0.3],
            'fp2': [0.4, 0.5, 0.6],
            'fp3': ['no', 'yes', 'yes']
        }),
        pd.DataFrame({
            'id': ['substrate1', 'substrate2', 'substrate3'],
            'fp2': [0.4, 0.5, 0.8],
            'fp3': ['no', 'yes', 'Yes'],
            'fp4': [0.7, 0.8, 0.9],
            'fp5': [1.0, 1.1, 1.2]
        })
    ]

    # Merge the DataFrames
    merged_df = merge_dfs(data, fingerprints, id_col='id', how='inner', desc_labels=None, duplicate_selection='last')

    # Check the shape of the merged DataFrame
    assert merged_df.shape == (3, 7)

    # Check the columns
    assert list(merged_df.columns) == ['id', 'yield', 'fp1_0', 'fp2_1', 'fp3_1', 'fp4_1', 'fp5_1']

    # Check the values
    expected_values = [
        ('substrate1', 10, 0.1, 0.4, 'no', 0.7, 1.0),
        ('substrate2', 20, 0.2, 0.5, 'yes', 0.8, 1.1),
        ('substrate3', 30, 0.3, 0.8, 'Yes', 0.9, 1.2)
    ]


    for i, (id_val, y, fp1_val, fp2_val, fp3_val, fp4_val, fp5_val) in enumerate(expected_values):
        assert merged_df.iloc[i].tolist() == [id_val, y, fp1_val, fp2_val, fp3_val, fp4_val, fp5_val]



#------------- Test function for merging data with duplicate_selection = 'mean' (work-in-progress)----------
def test_merge_data_mean():
    from src.preprocessing.merge_data import merge_dfs


    # Create mock DataFrames
    data = pd.DataFrame({
        'id': ['substrate1', 'substrate2', 'substrate3'],
        'yield': [10, 20, 30]
    })

    fingerprints = [
        pd.DataFrame({
            'id': ['substrate1', 'substrate2', 'substrate3'],
            'fp1': [0.1, 0.2, 0.3],
            'fp2': [0.4, 0.5, 0.6],
            'fp3': ['no', 'yes', 'yes']
        }),
        pd.DataFrame({
            'id': ['substrate1', 'substrate2', 'substrate3'],
            'fp2': [0.4, 0.5, 0.8],
            'fp3': ['no', 'yes', 'Yes'],
            'fp4': [0.7, 0.8, 0.9],
            'fp5': [1.0, 1.1, 1.2]
        })
    ]

    # Merge the DataFrames
    merged_df = merge_dfs(data, fingerprints, id_col='id', how='inner', desc_labels=None, duplicate_selection='first')

    # Check the shape of the merged DataFrame
    assert merged_df.shape == (3, 7)

    # Check the columns
    assert list(merged_df.columns) == ['id', 'yield', 'fp1_0', 'fp2_0', 'fp3_0', 'fp4_1', 'fp5_1']

    # Check the values
    expected_values = [
        ('substrate1', 10, 0.1, 0.4, 'no', 0.7, 1.0),
        ('substrate2', 20, 0.2, 0.5, 'yes', 0.8, 1.1),
        ('substrate3', 30, 0.3, 0.7, 'yes', 0.9, 1.2)
    ]


    for i, (id_val, y, fp1_val, fp2_val, fp3_val, fp4_val, fp5_val) in enumerate(expected_values):
        assert merged_df.iloc[i].tolist() == [id_val, y, fp1_val, fp2_val, fp3_val, fp4_val, fp5_val]
    