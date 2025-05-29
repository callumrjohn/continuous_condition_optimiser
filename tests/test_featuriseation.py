import sys
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem


def test_invalid_smiles():
    smiles = "INVALID"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is None  # RDKit returns None for bad SMILES

def test_gen_morgan_fps():
    from src.featurisation.morgan import gen_morgan_fps
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


def test_gen_rdkit_descriptors():
    from src.featurisation.rdkit import gen_rdkit_descriptors
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

