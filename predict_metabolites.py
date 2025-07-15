#pyright: basic

"""
Metabolite prediction script using GLORYxR.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from gloryxr.predictor import MetabolitePredictor
from gloryxr.utils import (load_models, load_reaction_subsets, load_sdf_data, 
                           print_summary, save_failed_molecules, save_predictions)


def predict_metabolites(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply metabolite prediction pipeline to molecules in the DataFrame.

    Args:
        df: DataFrame containing molecules (must have 'ROMol' column)

    Returns:
        Tuple of (predictions_df, failed_molecules_list)
    """
    print("Starting metabolite prediction pipeline...")

    # Load models reaction subsets
    try:
        models: dict[str, RandomForestClassifier] = load_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        return pd.DataFrame(), pd.DataFrame()
    try:
        reaction_subsets: dict[str, str] = load_reaction_subsets()
    except Exception as e:
        print(f"Error loading reaction subsets: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Initialize predictor
    try:
        predictor: MetabolitePredictor = MetabolitePredictor(models, reaction_subsets, strict_soms=False)
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Run predictions
    predictions, failed_molecules = predictor.predict_molecules(df)

    return predictions, failed_molecules


def main() -> int:
    """Main function to run metabolite prediction pipeline."""
    parser: ArgumentParser = ArgumentParser(description="Predict metabolites using GLORYxR")
    parser.add_argument("input_sdf", help="Path to input SDF file")
    parser.add_argument("output_folder", help="Path to output folder")

    args: Namespace = parser.parse_args()

    try:
        # Load data
        df: pd.DataFrame = load_sdf_data(sdf_path=args.input_sdf)

        # Run predictions
        predictions, failed_molecules = predict_metabolites(df)

        # Create output directory
        output_path: Path = Path(args.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results
        save_predictions(predictions, output_path)
        save_failed_molecules(failed_molecules, output_path)

        # Print summary
        print_summary(df, predictions, failed_molecules)

        print("Metabolite prediction completed successfully!")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(code=main())
