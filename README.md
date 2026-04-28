# Continuous Condition Optimiser

Screening framework for chemical featurisation methods from SMILES and machine learning model architectures to predict optimum ranges of continuous reaction parameters from experimental datasets.

## Features

- **Chemical Featurisation Methods**: Support for AQME, Custom, Fragprints, Mordred, Morgan fingerprints, and RDKit descriptors
- **Model Architectures**: Benchmark multiple ML models including Gaussian Process, MLP, Random Forest, SVM, and XGBoost
- **Preprocessing Pipeline**: Automated feature reduction, data transformation, and merging
- **Metrics & Validation**: Comprehensive performance evaluation and optimum region assessment

## Installation

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Generate Molecular Descriptors
Generate descriptors/fingerprints from SMILES strings using your preferred featurisation method:

```bash
python -m src.featurisation.morgan_gen
python -m src.featurisation.rdkit_gen
# Or other featurisation methods: aqme_gen, custom_gen, fragprint_gen, mordred_gen
```

Configure input/output paths and parameters in `configs/featurisation/*.yaml`.

### 2. Merge Descriptors with Experimental Data
Combine selected descriptor sets with experimental reaction data:

```bash
python -m src.preprocessing.merge_data
```

A GUI will prompt you to select which descriptor sets to combine. The script outputs a CSV with merged features and experimental values (e.g., yield, reagent equivalents).

### 3. Optional: Feature Reduction
Reduce feature dimensionality using PCA or RFE to improve model performance:

```bash
python -m src.preprocessing.feature_reduction
```

Configure the reduction method and parameters in `configs/preprocessing/feature_reduction.yaml`.

### 4. Extract Experimental Optimum Regions
Derive reference optimum regions from the experimental dataset for validation metrics:

```bash
python -m src.metrics.get_optimums
```

Analyses yield curves to identify high-performing parameter ranges. Output is used for custom metrics during model evaluation.

### 5. Validate Model
Evaluate a selected model architecture on a selected dataset using cross-validation:

```bash
python -m src.models.validate_model
```

A GUI will prompt you to select:
- Model architecture (GP, MLP, RF, SVM, XGBoost, etc.)
- Dataset

The validation method (leave-one-out or k-fold) is configured in `configs/models/validate_model.yaml`.

The script outputs:
- **Per-fold metrics**: CSV with MSE, MAE, R², and custom metrics for each fold
- **Overall evaluation**: Summary statistics aggregated across all folds
- **Validation log**: Updated tracking file of all validation runs

## Configuration

All pipeline stages are configured via YAML files in `configs/`:
- `base.yaml` - Global settings (paths, column names)
- `featurisation/*.yaml` - Descriptor generation parameters
- `preprocessing/*.yaml` - Data merging and feature reduction
- `metrics/*.yaml` - Threshold and interpolation settings
- `models/*.yaml` - Model selection and validation parameters

## Project Structure

- `src/` - Main pipeline code (featurisation, models, preprocessing, metrics)
- `configs/` - YAML configuration files for all pipeline stages
- `data/` - Input experimental data and processed features
- `outputs/` - Generated descriptors, trained models, and validation results
- `tests/` - Test suite for all modules

## License

BSD-3-Clause

