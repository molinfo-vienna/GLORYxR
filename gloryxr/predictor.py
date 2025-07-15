#pyright: basic

"""
Main class for metabolite prediction using GLORYxR.
"""

import pandas as pd
from rdkit.Chem.rdmolfiles import MolToSmiles
from rdkit.Chem.rdChemReactions import ChemicalReaction
from sklearn.ensemble import RandomForestClassifier

import gloryxr
from gloryxr.should_be_in_fame3r import Fame3RVectorizer
from gloryxr.utils import (create_failed_molecule_record,
                           extract_smiles_for_soms, format_predictions, reactions_to_table)



class MetabolitePredictor:
    """
    Main class for metabolite prediction using GLORYxR.
    """

    def __init__(
        self, models: dict[str, RandomForestClassifier], reaction_subsets: dict[str, str], strict_soms: bool = False
    ) -> None:
        """
        Initialize the metabolite predictor.

        Args:
            models: Dictionary of loaded models
            reaction_subsets: Dictionary mapping reaction names to subsets
            strict_soms: Whether to use strict SOM validation
        """
        self.models: dict[str, RandomForestClassifier] = models
        self.reaction_subsets: dict[str, str] = reaction_subsets
        self.strict_soms: bool = strict_soms

        # Initialize vectorizer
        self.vectorizer = Fame3RVectorizer().fit()

        # Initialize reactor
        try:
            self.reactor = gloryxr.Reactor(strict_soms=strict_soms)
            print(
                f"Initialized reactor with {len(self.reactor.abstract_reactions)} reaction rules"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize reactor: {e}")

    def predict_single_molecule(
        self, mol, molecule_id: int, parent_name: str
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Predict metabolites for a single molecule.

        Args:
            mol: RDKit molecule object
            molecule_id: Index of the molecule
            parent_name: Name/ID of the parent molecule

        Returns:
            Tuple of (predictions_df, failed_record_df) - one will be None
        """
        if mol is None:
            failed_record = create_failed_molecule_record(
                molecule_id, parent_name, failure_reason="invalid_mol_object"
            )
            return None, failed_record

        try:
            # Generate concrete reactions
            reactions: list[ChemicalReaction] = self.reactor.react_one(mol)
            concrete_reactions = reactions_to_table(reactions).pipe(
                lambda df: df.assign(
                    Subset=df.Reaction.map(
                        lambda name: self.reaction_subsets.get(name, "Unknown")
                    ),
                )
            )
        except Exception as e:
            failed_record = create_failed_molecule_record(
                molecule_id,
                parent_name,
                failure_reason=f"reaction_generation_error: {str(e)}",
                parent_smiles=MolToSmiles(mol) if mol else None,
            )
            return None, failed_record

        if concrete_reactions.empty:
            failed_record = create_failed_molecule_record(
                molecule_id,
                parent_name,
                failure_reason="no_reactions_found",
                parent_smiles=MolToSmiles(mol) if mol else None,
            )
            return None, failed_record

        try:
            predictions = self.apply_prediction_pipeline(
                concrete_reactions, mol, molecule_id, parent_name
            )
            return predictions, None
        except Exception as e:
            failed_record = create_failed_molecule_record(
                molecule_id,
                parent_name,
                failure_reason=f"prediction_pipeline_error: {str(e)}",
                parent_smiles=MolToSmiles(mol) if mol else None,
            )
            return None, failed_record

    def apply_prediction_pipeline(
        self, concrete_reactions: pd.DataFrame, mol, molecule_id: int, parent_name: str
    ) -> pd.DataFrame:
        """
        Apply the complete prediction pipeline to concrete reactions.

        Args:
            concrete_reactions: DataFrame with concrete reactions
            mol: Original molecule
            molecule_id: Molecule index
            parent_name: Molecule name

        Returns:
            DataFrame with predictions
        """
        # Extract SOMs
        som_predictions = concrete_reactions.pipe(
            lambda df: df.assign(SOM=df.Educt.map(extract_smiles_for_soms))
        )

        # Explode SOMs
        exploded_predictions = som_predictions.explode("SOM")

        if exploded_predictions.empty:
            raise ValueError("No SOMs extracted")

        # Generate descriptors
        descriptor_predictions = exploded_predictions.pipe(
            lambda df: df.assign(Descriptors=df.SOM.map(self.vectorizer.transform_one))
        )

        # Apply model predictions
        predictions = descriptor_predictions.pipe(
            lambda df: df.assign(
                Score=df.apply(
                    lambda row: (
                        self.models[row.Subset].predict_proba([row.Descriptors])[0][-1]
                        if row.Subset in self.models
                        else 0.0
                    ),
                    axis=1,
                )
            )
        )

        # Convert mol objects to SMILES and rename columns
        predictions["parent_smiles"] = predictions["Educt"].apply(MolToSmiles)
        predictions["metabolite_smiles"] = predictions["Product"].apply(MolToSmiles)

        # Remove unnecessary columns
        predictions = predictions.drop(
            ["Educt", "Product", "Descriptors"], axis=1, errors="ignore"
        )

        # Add molecule identifier
        predictions["parent_name"] = parent_name

        return predictions

    def predict_molecules(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict metabolites for all molecules in the DataFrame.

        Args:
            df: DataFrame containing molecules (must have 'ROMol' column)

        Returns:
            Tuple of (predictions_df, failed_molecules_df)
        """
        print(f"Processing {len(df)} molecules...")

        results: list[tuple[pd.DataFrame | None, pd.DataFrame | None]] = [
            self.predict_single_molecule(
                mol=row["ROMol"], 
                molecule_id=i, 
                parent_name=str(row.get(key="ID", default=f"molecule_{i}"))
            )
            for i, (_, row) in enumerate(df.iterrows())
        ]

        predictions: list[pd.DataFrame] = [r[0] for r in results if r[0] is not None]
        fails: list[pd.DataFrame] = [r[1] for r in results if r[1] is not None]

        df_predictions: pd.DataFrame = format_predictions(predictions=pd.concat(predictions, ignore_index=True)) if predictions else pd.DataFrame()
        df_fails: pd.DataFrame = pd.concat(fails, ignore_index=True) if fails else pd.DataFrame()

        return df_predictions, df_fails
