# pyright: basic

"""
Main class for metabolite prediction using GLORYxR.
"""

import pandas as pd
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import GetMolFrags
from sklearn.ensemble import RandomForestClassifier

import gloryxr
from gloryxr.should_be_in_fame3r import Fame3RVectorizer
from gloryxr.utils import (clean_smiles, create_failed_molecule_record,
                           extract_smiles_for_soms, get_largest_fragment, reactions_to_table)


class MetabolitePredictor:
    """
    Main class for metabolite prediction using GLORYxR.
    """

    def __init__(
        self,
        models: dict[str, RandomForestClassifier],
        reaction_subsets: dict[str, str],
        strict_soms: bool = False,
    ) -> None:
        """
        Initialize the metabolite predictor.

        Args:
            models: Dictionary of loaded models
            reaction_subsets: Dictionary mapping reaction names to rule subsets
            strict_soms: Whether to use strict SOMs
        """
        self.models = models
        self.reaction_subsets = reaction_subsets
        self.strict_soms = strict_soms
        self.vectorizer = Fame3RVectorizer().fit()

        # Initialize reactor
        self.reactor = gloryxr.Reactor(strict_soms=strict_soms)
        print(
            f"Initialized reactor with {len(self.reactor.abstract_reactions)} reaction rules"
        )

    def predict_molecules(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict metabolites for all molecules in the DataFrame.

        Args:
            df: DataFrame containing molecules (must have 'ROMol' column)

        Returns:
            Tuple of (predictions_df, failed_molecules_df)
        """
        print(f"Processing {len(df)} molecules...")

        predictions_list = []
        failed_list = []

        for molecule_id, (_, row) in enumerate(df.iterrows()):
            mol = row["ROMol"]
            parent_name = str(row.get(key="ID", default=f"molecule_{molecule_id}"))

            result = self._process_molecule(mol, parent_name)

            if result is None:
                failed_list.append(
                    create_failed_molecule_record(
                        molecule_id, parent_name, failure_reason="processing_error"
                    )
                )
            elif isinstance(result, pd.DataFrame) and len(result) == 0:
                failed_list.append(
                    create_failed_molecule_record(
                        molecule_id,
                        parent_name,
                        failure_reason="no_predictions_generated",
                        parent_smiles=MolToSmiles(mol) if mol is not None else None,
                    )
                )
            else:
                predictions_list.append(result)

        # Combine results
        predictions_df = (
            pd.concat(predictions_list, ignore_index=True)
            if predictions_list
            else pd.DataFrame()
        )
        failed_df = (
            pd.concat(failed_list, ignore_index=True) if failed_list else pd.DataFrame()
        )

        # Clean up smiles
        predictions_df["parent_smiles"] = predictions_df["parent_smiles"].apply(
            clean_smiles
        )
        predictions_df["metabolite_smiles"] = predictions_df["metabolite_smiles"].apply(
            clean_smiles
        )

        # Filter out duplicated predictions
        predictions_df = pd.DataFrame(
            predictions_df.sort_values(by="Score", ascending=False)
            .groupby(by=["metabolite_smiles"], sort=False, as_index=False)
            .first()
            .reset_index(drop=True)[
                [
                    "parent_name",
                    "parent_smiles",
                    "metabolite_smiles",
                    "Reaction",
                    "Subset",
                    "SOM",
                    "Score",
                ]
            ]
        )

        # If a "metabolite" consists in multiple fragments, only keep the largest fragment
        predictions_df["metabolite_smiles"] = predictions_df["metabolite_smiles"].apply(get_largest_fragment)

        # Filter out predictions with less than 3 heavy atoms
        mask = predictions_df["metabolite_smiles"].apply(
            lambda x: MolFromSmiles(x).GetNumHeavyAtoms() >= 3
        )
        predictions_df = pd.DataFrame(predictions_df[mask])

        # Format predictions dataframe (lowercase column names)
        predictions_df.rename(
            columns={
                "Subset": "rule_subset",
                "Reaction": "reaction",
                "Score": "score",
                "SOM": "som",
            },
            inplace=True,
        )

        # Resort predictions according to parent compound and score
        predictions_df = predictions_df.sort_values(by=["parent_smiles", "score"], ascending=[True, False])

        return predictions_df, failed_df

    def _process_molecule(self, mol, parent_name: str) -> pd.DataFrame | None:
        """
        Process a single molecule and return predictions or None if failed.

        Args:
            mol: RDKit molecule object
            parent_name: Name/ID of the molecule

        Returns:
            DataFrame with predictions or None if failed
        """
        if mol is None:
            return None

        try:
            reactions = self.reactor.react_one(mol)
            if not reactions:
                return pd.DataFrame()

            reactions_df = reactions_to_table(reactions)
            reactions_df["Subset"] = reactions_df["Reaction"].map(
                lambda name: self.reaction_subsets.get(name, "Unknown")
            )

            return self._generate_predictions(reactions_df, parent_name)

        except Exception as e:
            print(f"Error processing molecule {parent_name}: {e}")
            return None

    def _generate_predictions(
        self, reactions_df: pd.DataFrame, parent_name: str
    ) -> pd.DataFrame:
        """
        Generate predictions from reactions DataFrame.

        Args:
            reactions_df: DataFrame with reactions
            parent_name: Name of the parent molecule

        Returns:
            DataFrame with predictions
        """
        # Extract SOMs and explode
        reactions_df["SOM"] = reactions_df["Educt"].map(extract_smiles_for_soms)
        exploded_df = reactions_df.explode("SOM")

        if len(exploded_df) == 0:
            return pd.DataFrame()

        # Generate descriptors
        exploded_df["Descriptors"] = exploded_df["SOM"].map(
            self.vectorizer.transform_one
        )

        # Apply model predictions
        exploded_df["Score"] = exploded_df.apply(
            lambda row: self._get_prediction_score(row), axis=1
        )

        # Convert to final format
        predictions = exploded_df.copy()
        predictions["parent_smiles"] = predictions["Educt"].apply(MolToSmiles)
        predictions["metabolite_smiles"] = predictions["Product"].apply(MolToSmiles)
        predictions["parent_name"] = parent_name

        # Keep only necessary columns and ensure it's a DataFrame
        columns_to_keep = [
            "parent_name",
            "parent_smiles",
            "metabolite_smiles",
            "Reaction",
            "Subset",
            "SOM",
            "Score",
        ]
        result_df = predictions.loc[:, columns_to_keep].copy()
        return result_df

    def _get_prediction_score(self, row) -> float:
        """Get prediction score for a single row."""
        subset = row["Subset"]
        descriptors = row["Descriptors"]
        priority = row["Priority"]

        if priority == "common":
            priority_factor = 1.
        elif priority == "uncommon":
            priority_factor = 0.2
        else:
            raise ValueError(f"Invalid priority: {priority}")

        if subset in self.models:
            som_probability = self.models[subset].predict_proba([descriptors])[0][-1]
            return som_probability * priority_factor
        return 0.0
