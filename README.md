# Halogenation Condition Prediction

This research project aims to predict optimal acidic reaction conditions for the C(sp<sup>2</sup>)–H chlorination, bromination, and iodination of pharmaceutical compounds using machine learning.

## Overview

Given:
- The SMILES representation of a pharmaceutical compound
- The desired halogenation reaction type

The model predicts:
- Optimum reaction conditions with respect to acitity (the number of TFA equivalents in HFIP at 0.1 M)

## Motivation

Halogenation is a key transformation in pharmaceutical synthesis. Using a high-throughput workflow, we have generate a dataset comprising 31 pharmaceutical compounds with three halogenations performed at a range of acidities. This project leverages machine learning and the dataset to predict high-yeidling conditions for a given unscreened pharmacetucial halogenation, reducing the need for acidity screening.

## Features

- Input: SMILES string of a pharmaceutical, halogenation type
- Output: Predicted reaction conditions
- Trained on curated reaction datasets

## Usage

1. Prepare input data (SMILES, reaction type)
2. Run the prediction model
3. Obtain recommended conditions

## Installation

```bash
git clone https://github.com/callumrjohn/halogenation_condition_prediction.git
cd halogenation_condition_prediction
# Install dependencies
pip install -r requirements.txt
```

## Running Predictions

```python
from predictor import predict_conditions

smiles = "CC1=C(C2=C(N1C(C3=CC=C(Cl)C=C3)=O)C=CC(OC)=C2)CC(O)=O" # Indomethacin
reaction_type = "iodination"
conditions = predict_conditions(smiles, reaction_type)
print(conditions)
```

## Data

- The dataset was generated experimentally using an automated screening platform developed alongside this researach, comprising data from the halogenation of 31 pharmaceutical compounds using 3 halogenating 
- Preprocessing scripts included

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

BSD-3-Clause

## Contact

