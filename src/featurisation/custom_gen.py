import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from src.utils.config import load_config

'''SULFUR NUCLEOPHILES-----------------------------------------------'''

def has_thiol(mol):
    """Check if a molecule contains a thiol group (-SH)."""
    thiol_smarts = Chem.MolFromSmarts('[SX2H]')
    return mol.HasSubstructMatch(thiol_smarts)

def has_thioether(mol):
    """Check if a molecule contains a thioether group (R-S-R')."""
    thioether_smarts = Chem.MolFromSmarts('[S&D2;!a;$(S-[C;!$(C=O)])][C;!$(C=O)]') # does NOT include thioethers with carbonyls or aromatic sulfures
    return mol.HasSubstructMatch(thioether_smarts)

def has_thioester(mol):
    """Check if a molecule contains a thioester group (R-S-C(=O)-R')."""
    thioester_smarts = Chem.MolFromSmarts('[C](=O)[S][#6]')
    return mol.HasSubstructMatch(thioester_smarts)

def has_thiocarboxylic_acid(mol):
    """Check if a molecule contains a thiocarboxylic acid group (R-S-C(=S)-OH)."""
    thiocarboxylic_acid_smarts = Chem.MolFromSmarts('[C](=O)[SX2H]')
    return mol.HasSubstructMatch(thiocarboxylic_acid_smarts)

def has_sulfoxide(mol):
    """Check if a molecule contains a sulfoxide group (R-S(=O)-R')."""
    sulfoxide_smarts = Chem.MolFromSmarts('[S](=O)[#6]')
    return mol.HasSubstructMatch(sulfoxide_smarts)

'''OTHER INCOMPATABLE MOIETIES?--------------------------------------'''

def has_alkene(mol):
    """Check if a molecule contains an alkene group (C=C)."""
    alkene_smarts = Chem.MolFromSmarts('C=C')
    return mol.HasSubstructMatch(alkene_smarts)

def has_alkyne(mol):
    """Check if a molecule contains an alkyne group (C#C)."""
    alkyne_smarts = Chem.MolFromSmarts('C#C')
    return mol.HasSubstructMatch(alkyne_smarts)

def has_tertiary_alcohol(mol):
    """Check if a molecule contains a tertiary alcohol group."""
    tertiary_alcohol_smarts = Chem.MolFromSmarts('[CX4](O)([CX4])([CX4])[CX4]')
    return mol.HasSubstructMatch(tertiary_alcohol_smarts)

def has_glycosamine(mol):
    """Check if a molecule contains a glycosamine group."""
    glycosamine_smarts = Chem.MolFromSmarts('[C;!$(C=O)]([O;!$(C=O)])([N;!$(C=O)])')
    return mol.HasSubstructMatch(glycosamine_smarts)

def has_glycoside(mol):
    """Check if a molecule contains a glycoside group."""
    glycoside_smarts = Chem.MolFromSmarts('[C;!$(C=O)]([O;!$(C=O)])([O;!$(C=O)])')
    return mol.HasSubstructMatch(glycoside_smarts)



def gen_custom_descriptors(smiles):
    """Generate custom features based on specific moieties in the SMILES strings."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

        
    custom_features = {}

    custom_features['has_thiol'] = has_thiol(mol)
    custom_features['has_thioether'] = has_thioether(mol)
    custom_features['has_thioester'] = has_thioester(mol)
    custom_features['has_thiocarboxylic_acid'] = has_thiocarboxylic_acid(mol)
    custom_features['has_sulfoxide'] = has_sulfoxide(mol)
    custom_features['has_alkene'] = has_alkene(mol)
    custom_features['has_alkyne'] = has_alkyne(mol)
    custom_features['has_tertiary_alcohol'] = has_tertiary_alcohol(mol)
    custom_features['has_glycosamine'] = has_glycosamine(mol)
    custom_features['has_glycoside'] = has_glycoside(mol)

    return custom_features
