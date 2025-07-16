#pyright: basic

"""
Metabolite prediction using GLORYxR.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from gloryxr.predictor import MetabolitePredictor
from gloryxr.utils import (load_models, load_reaction_subsets, load_sdf_data, 
                           print_summary, save_failed_molecules, save_predictions, standardize_molecules)


def load_prediction_components() -> tuple[dict[str, RandomForestClassifier], dict[str, str]]:
    """
    Load models and reaction subsets.
    
    Returns:
        Tuple of (models, reaction_subsets)
        
    Raises:
        RuntimeError: If components cannot be loaded
    """
    try:
        models = load_models()
        reaction_subsets = load_reaction_subsets()
        return models, reaction_subsets
    except Exception as e:
        raise RuntimeError(f"Failed to load prediction components: {e}")


def run_prediction_pipeline(input_sdf: str, output_folder: str) -> int:
    """
    Run the complete metabolite prediction pipeline.
    
    Args:
        input_sdf: Path to input SDF file
        output_folder: Path to output folder
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        print("Loading data...")
        df = load_sdf_data(input_sdf)
        
        print("Standardizing molecules...")
        df, standardization_failed = standardize_molecules(df)
        
        if len(df) == 0:
            print("No molecules passed standardization. Exiting.")
            return 1
        
        print("Loading prediction models...")
        models, reaction_subsets = load_prediction_components()
        
        print("Initializing predictor...")
        predictor = MetabolitePredictor(models, reaction_subsets, strict_soms=True)
        
        print("Running predictions...")
        predictions, prediction_failed = predictor.predict_molecules(df)
        
        # Combine failed molecules from standardization and prediction procedures
        failed_molecules = pd.concat([standardization_failed, prediction_failed], ignore_index=True) if not standardization_failed.empty or not prediction_failed.empty else pd.DataFrame()
        
        print("Saving results...")
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        save_predictions(predictions, output_path)
        save_failed_molecules(failed_molecules, output_path)
        
        # Print summary
        print_summary(df, predictions, failed_molecules)
        
        print("Metabolite prediction completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


def main() -> int:
    """Main function to run metabolite prediction pipeline."""
    parser = ArgumentParser(description="Predict metabolites using GLORYxR")
    parser.add_argument("input_sdf", help="Path to input SDF file")
    parser.add_argument("output_folder", help="Path to output folder")

    args: Namespace = parser.parse_args()
    return run_prediction_pipeline(args.input_sdf, args.output_folder)


if __name__ == "__main__":
    exit(code=main())
