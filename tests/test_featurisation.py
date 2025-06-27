import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem


#----------------- Test function for invalid SMILES ----------------
def test_invalid_smiles():
    smiles = "INVALID"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is None  # RDKit returns None for bad SMILES


#----------------- Test function for generating Morgan fingerprints ----------------
def test_gen_morgan_fps():
    from src.featurisation.morgan_gen import gen_morgan_fps
    # Mock data
    df = pd.DataFrame({
        "id": ['Caffeine', 
               'Lamivudine', 
               'Invalid'
               ],
        "smiles": ["O=C(N(C1=O)C)N(C2=C1N(C=N2)C)C",
                   "NC1=NC(N([C@@H]2CS[C@@H](O2)CO)C=C1)=O",
                   "INVALID"
                   ]
    })
    fp_df, failed = gen_morgan_fps(df, "smiles", "id", radius=2, nBits=16)

    assert fp_df.shape == (3, 17)  # 16 bits + 1 id
    
    assert failed == ['Invalid']  # Invalid SMILES should be in failed
    
    assert fp_df["id"].tolist() == ['Caffeine', 'Lamivudine', 'Invalid'] # IDs should match input


#----------------- Test function for generating RDKit descriptors ----------------
def test_gen_rdkit_descriptors():
    from src.featurisation.rdkit_gen import gen_rdkit_descriptors
    # Mock data
    df = pd.DataFrame({
        "id": ['Caffeine', 
               'Lamivudine', 
               'Invalid'
               ],
        "smiles": ["O=C(N(C1=O)C)N(C2=C1N(C=N2)C)C",
                   "NC1=NC(N([C@@H]2CS[C@@H](O2)CO)C=C1)=O",
                   "INVALID"
                   ]
    })
    
    rdk_df = gen_rdkit_descriptors(df, "smiles", "id", missingVal=np.nan, silent=True)

    assert rdk_df.shape[1] > 1  # Should have more than just the ID column

    assert 'MolWt' in rdk_df.columns  # Check if a common descriptor is present
    
    assert rdk_df.iloc[2, 1:].isnull().all()  # Invalid SMILES should have NaN for all descriptors

    assert rdk_df["id"].tolist() == ['Caffeine', 'Lamivudine', 'Invalid']  # IDs should match input


#----------------- Test function for generating Mordred fingerprints --------------

def test_gen_mordred_descriptors():
    from src.featurisation.mordred_gen import gen_mordred_descriptors
    # Mock data
    df = pd.DataFrame({
        "id": ['Caffeine', 
               'Lamivudine', 
               'Invalid'
               ],
        "smiles": ["O=C(N(C1=O)C)N(C2=C1N(C=N2)C)C",
                   "NC1=NC(N([C@@H]2CS[C@@H](O2)CO)C=C1)=O",
                   "INVALID"
                   ]
    })
    
    mord_df = gen_mordred_descriptors(df, "smiles", "id", ignore_3D = True, missingVal = np.nan)

    assert mord_df.shape[1] == 1614  # ID column + 1613 Mordred descriptors

    assert 'MW' in mord_df.columns  # Check if a common descriptor is present
    
    assert mord_df.iloc[2, 1:].isnull().all()  # Invalid SMILES should have NaN for all descriptors

    assert mord_df["id"].tolist() == ['Caffeine', 'Lamivudine', 'Invalid']  # IDs should match input


#----------------- Test function for generating custom fingerprints ---------------

def test_get_custom_descriptors():
    from src.featurisation.custom_gen import gen_custom_descriptors
    # Mock data
    df = pd.DataFrame({
        "id": ['Caffeine',
               'pyrimidin-2(1H)-one', 
               'Lamivudine',
               'Tetracycline',
               'Propane',
               'Thiophene',
               '3,6-dihydro-2H-thiopyuran',
               'Propan-2-ol',
               'Ethanethioic acid',
               '(2S,3S)-2-aminotetrahydro-2H-pyran-3-ol',
               'S-ethyl 4-hydroxybut-3-ynethioate',
               '1-(methylthio)but-3-en-2-one',
               '(2S,3S)-2-methoxytetrahydro-2H-pyran-3-ol',
               ],
        "smiles": ['O=C(N(C1=O)C)N(C2=C1N(C=N2)C)C',
                   'O=C1N=CC=CN1',
                   'NC1=NC(N([C@@H]2CS[C@@H](O2)CO)C=C1)=O',
                   'OC1=CC=CC2=C1C(C3=C(O)[C@@](C(C(C(N)=O)=C(O)[C@H]4N(C)C)=O)(O)C4CC3[C@@]2(O)C)=O',
                   'CCC',
                   'C1=CC=CS1',
                   'C1C=CCCS1',
                   'CC(O)C',
                   'CC(=O)S',
                   'N[C@@H]1[C@@H](O)CCCO1',
                   'OC#CCC(SCC)=O',
                   'O=C(CSC)C=C',
                   'CO[C@@H]1[C@@H](O)CCCO1'
                   ]
    })
    
    descriptors = pd.DataFrame(columns=['smiles', 'has_thiol', 'has_thioether', 'has_thioester',
                                       'has_thiocarboxylic_acid', 'has_sulfoxide',
                                       'has_alkene', 'has_alkyne', 'has_tertiary_alcohol',
                                       'has_glycosamine', 'has_glycoside'])
    for smiles in df['smiles']:
        custom_features = gen_custom_descriptors(smiles)
        new_row = pd.DataFrame([{'smiles': smiles, **custom_features}])
        descriptors = pd.concat([descriptors, new_row], ignore_index=True)
    
    # Check the shape of the descriptors DataFrame
    assert descriptors.shape == (13, 11)  # 10 features + 1 smiles column

    descriptor_values = descriptors.iloc[:, 1:].values

    expected_values = [[False, False, False, False, False, False, False, False, False, False],
                       [False, False, False, False, False, False, False, False, False, False],
                       [False, True, False, False, False, False, False, False, True, False],
                       [False, False, False, False, False, True, False, True, False, False],
                       [False, False, False, False, False, False, False, False, False, False],
                       [False, False, False, False, False, False, False, False, False, False],
                       [False, True, False, False, False, True, False, False, False, False],
                       [False, False, False, False, False, False, False, False, False, False],
                       [False, False, False, True, False, False, False, False, False, False],
                       [False, False, False, False, False, False, False, False, True, False],
                       [False, False, True, False, False, False, True, False, False, False],
                       [False, True, False, False, False, True, False, False, False, False],
                       [False, False, False, False, False, False, False, False, False, True],
    ]

    # Check if the descriptor values match the expected values
    for i, expected in enumerate(expected_values):
        print(f"Descriptor values for {df['id'][i]}: {descriptor_values[i]}")
        print(f"Expected values: {expected}")
        assert np.array_equal(descriptor_values[i], expected)
        

#----------------- Test function for generating custom fingerprints ----------------
def test_gen_custom_descriptors():
    from src.featurisation.custom_gen import gen_custom_descriptors
    # Mock data
    df = pd.DataFrame({
        "id": ['Caffeine', 
               'Lamivudine', 
               'Invalid'
               ],
        "smiles": ["O=C(N(C1=O)C)N(C2=C1N(C=N2)C)C",
                   "NC1=NC(N([C@@H]2CS[C@@H](O2)CO)C=C1)=O",
                   "INVALID"
                   ]
    })
    
    custom_df, failed = gen_custom_descriptors(df, "smiles", "id")

    assert custom_df.shape[1] > 1  # Should have more than just the ID column

    assert 'has_sulfoxide' in custom_df.columns  # Check if a common descriptor is present
    
    assert failed == ['Invalid']
    
    assert custom_df.iloc[2, 1:].isnull().all()  # Invalid SMILES should have NaN for all descriptors

    assert custom_df["id"].tolist() == ['Caffeine', 'Lamivudine', 'Invalid']  # IDs should match input


#----------------- Test function for generating processed AQME descriptors ----------------
def test_seperate_atomic_descriptors():
    from src.featurisation.aqme_gen import separate_atomic_descriptors
    # Mock data
    atom_df = pd.DataFrame({
        "substrate_id": ['Caffeine', 'Lamivudine'],
        "1": [[1.0, 2.0], [3.0, 4.0]],
        "2": [[5.0, 6.0], [7.0, 8.0]],
        "3": [9.0, 12.0],
    })
    
    id_col = 'substrate_id'
    
    atom_df, mol_df = separate_atomic_descriptors(atom_df, id_col=id_col)

    assert mol_df.shape[1] == 2  # Should have substrate_id and mol column
    assert atom_df.shape[1] == 3  # Should have substrate_id and two atom columns

    assert mol_df["substrate_id"].tolist() == ['Caffeine', 'Lamivudine']
    assert atom_df["substrate_id"].tolist() == ['Caffeine', 'Lamivudine']

def test_get_abs_minmax_df():
    from src.featurisation.aqme_gen import get_abs_minmax_df
    # Mock data
    atom_df = pd.DataFrame({
        "substrate_id": ['Caffeine', 'Lamivudine'],
        "1": [[1.0, 2.0], [3.0, 4.0]],
        "2": [[5.0, 6.0], [7.0, 8.0]],
        "3": [9.0, 12.0],
    })
    
    id_col = 'substrate_id'
    
    abs_minmax_df = get_abs_minmax_df(atom_df, id_col=id_col)

    assert abs_minmax_df.shape[1] == 5  # Should have substrate_id and min/max for each column

    assert abs_minmax_df["substrate_id"].tolist() == ['Caffeine', 'Lamivudine']
    assert abs_minmax_df["1_min"].tolist() == [1.0, 3.0]
    assert abs_minmax_df["1_max"].tolist() == [2.0, 4.0]
    assert abs_minmax_df["2_min"].tolist() == [5.0, 7.0]
    assert abs_minmax_df["2_max"].tolist() == [6.0, 8.0]
    assert abs_minmax_df["3_min"].tolist() == [9.0, 12.0]
    assert abs_minmax_df["3_max"].tolist() == [9.0, 12.0]