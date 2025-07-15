# Metabolite Prediction with GLORYxR

This script provides a command-line interface for predicting metabolites using the GLORYxR framework.

## Usage

```bash
python predict_metabolites.py <input_sdf> <output_folder>
```

### Arguments

- `input_sdf`: Path to the input SDF file containing molecules to predict metabolites for
- `output_folder`: Path to the output folder where results will be saved

### Example

```bash
python predict_metabolites.py GLORYx_data/test/gloryx_test_dataset.sdf GLORYx_data/test/predictions/
```

## Input Format

The input SDF file should contain:
- Valid molecular structures in SDF format

## Output

The script generates:
- `metabolite_predictions.csv`: CSV file containing all metabolite predictions with scores
- `failed_molecules.csv`: CSV file containing molecules that failed during prediction with failure reasons
- Console output showing:
  - Progress during processing
  - Summary statistics

## Output Columns

The prediction CSV contains (in this order):
- `parent_name`: Name/ID of the parent molecule (from SDF ID field or auto-generated)
- `parent_smiles`: SMILES of the parent molecule
- `metabolite_smiles`: SMILES of the predicted metabolite
- `reaction`: Name of the reaction applied
- `rule_subset`: Reaction subset category
- `som`: Site of metabolism
- `score`: Prediction score (probability)

## Failed Molecules CSV

The `failed_molecules.csv` file contains:
- `molecule_id`: Index of the failed molecule
- `parent_name`: Name/ID of the parent molecule
- `failure_reason`: Description of why the molecule failed
- `parent_smiles`: SMILES of the parent molecule (if available)

Common failure reasons:
- `invalid_mol_object`: Molecule could not be parsed from SDF
- `no_reactions_found`: No metabolic reactions were identified
- `no_soms_extracted`: No sites of metabolism were found
- `reaction_generation_error`: Error during reaction generation
- `prediction_pipeline_error`: Error during descriptor generation or model prediction
- `unexpected_error`: Other unexpected errors

## Modular Structure

The codebase is organized into modular components:

### `gloryxr/predictor.py`
Main class for metabolite prediction using GLORYxR.

### `gloryxr/reaction.py`
Functions to convert abstract reactions to concrete reactions.

### `gloryxr/reactor.py`
Core chemical reaction processing engine.

### `gloryxr/should_be_in_fane3r.py`
Functonalities that will be put in FAME3R at the end.

### `gloryxr/som.py`
Functions to annotate the educt molecule with SOM indices.

### `gloryxr/utils.py`
Various utility functions.

### `predict_metabolites.py`
Main script that orchestrates the prediction pipeline.
