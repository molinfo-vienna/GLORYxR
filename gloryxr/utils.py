# pyright: basic

"""
Utility functions for GLORYxR metabolite prediction.
"""

import csv
from pathlib import Path

import joblib
import pandas as pd
from molvs import Standardizer
from rdkit.Chem.PandasTools import LoadSDF
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import GetMolFrags
from sklearn.ensemble import RandomForestClassifier


def clean_smiles(smiles: str) -> str:
    """
    Clean up SMILES string.
    """
    mol: Mol = MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"Cannot parse SMILES: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return MolToSmiles(mol, canonical=True, ignoreAtomMapNumbers=True)


def reactions_to_table(reactions: list[ChemicalReaction]) -> pd.DataFrame:
    """
    Convert chemical reactions to a pandas DataFrame.

    Args:
        reactions: Iterable of chemical reactions

    Returns:
        DataFrame with Educt, Product, and Reaction columns
    """
    return pd.DataFrame(
        [
            {
                "Educt": reaction.GetReactants()[0],
                "Product": reaction.GetProducts()[0],
                "Reaction": (
                    reaction.GetProp("_Name") if reaction.HasProp("_Name") else None
                ),
            }
            for reaction in reactions
        ]
    ).rename_axis("ID")


def extract_smiles_for_soms(mol: Mol) -> list[str]:
    """
    Extract SMILES strings for sites of metabolism from a molecule.

    Args:
        mol: RDKit molecule object

    Returns:
        List of SMILES strings for each SOM
    """
    mapno_to_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() != 0
    }

    results = []
    for mapno in sorted(mapno_to_idx.keys()):
        single_mol = Mol(mol)
        for atom in single_mol.GetAtoms():
            if mapno_to_idx[mapno] != atom.GetIdx():
                atom.SetAtomMapNum(0)

        results.append(single_mol)

    return [MolToSmiles(mol) for mol in results]


def load_models(models_dir: str = "models") -> dict[str, RandomForestClassifier]:
    """
    Load trained models from joblib files.

    Args:
        models_dir: Directory containing .joblib model files

    Returns:
        Dictionary mapping model names to loaded models
    """
    models_path: Path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    models: dict[str, RandomForestClassifier] = {}
    for model_file in models_path.glob(pattern="*.joblib"):
        model_name: str = model_file.stem
        try:
            models[model_name] = joblib.load(filename=model_file)
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")

    return models


def load_reaction_subsets(
    rules_file: str = "./gloryxr/rules_data/gloryx_reactionrules.csv",
) -> dict[str, str]:
    """
    Load reaction subset mappings from CSV file.

    Args:
        rules_file: Path to the reaction rules CSV file

    Returns:
        Dictionary mapping reaction names to subset names
    """
    rules_path: Path = Path(rules_file)
    if not rules_path.exists():
        raise FileNotFoundError(f"Reaction rules file not found: {rules_file}")

    reaction_subsets: dict[str, str] = {}
    with open(file=rules_path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reaction_subsets[row["Reaction name"]] = row["Name of rule subset"]

    return reaction_subsets


def load_sdf_data(sdf_path: str) -> pd.DataFrame:
    """
    Load SDF file into a Pandas DataFrame with RDKit mol objects.

    Args:
        sdf_path: Path to the SDF file

    Returns:
        DataFrame containing the molecules and their properties

    Raises:
        ValueError: If SDF file is empty
        KeyError: If SDF file does not contain 'ROMol' column
    """
    sdf_file: Path = Path(sdf_path)
    if not sdf_file.exists():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    print(f"Loading SDF file: {sdf_path}")

    df: pd.DataFrame = LoadSDF(filename=sdf_path, removeHs=False, sanitize=True)

    if df is df.empty:
        raise ValueError(
            "Input DataFrame is empty. Check that the SD file contains valid molecules."
        )

    if "ROMol" not in df.columns:
        raise KeyError(
            "DataFrame does not contain 'ROMol' column. Check that the SD file contains valid molecules."
        )

    print(f"Loaded {len(df)} molecules from SDF file")
    return df


def save_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
    filename: str = "metabolite_predictions.csv",
) -> None:
    """
    Save predictions DataFrame to CSV file.

    Args:
        predictions: DataFrame containing predictions
        output_path: Output directory path
        filename: Name of the output file
    """
    if not predictions.empty:
        output_file: Path = output_path / filename
        predictions.to_csv(output_file, index=False)
        print(f"Saved {len(predictions)} predictions to {output_file}")


def save_failed_molecules(
    failed_molecules: pd.DataFrame,
    output_path: Path,
    filename: str = "failed_molecules.csv",
) -> None:
    """
    Save failed molecules DataFrame to CSV file.

    Args:
        failed_molecules: DataFrame containing failed molecule info
        output_path: Output directory path
        filename: Name of the output file
    """
    if not failed_molecules.empty:
        failed_file: Path = output_path / filename
        failed_molecules.to_csv(failed_file, index=False)
        print(f"Saved {len(failed_molecules)} failed molecules to {failed_file}")


def print_summary(
    df: pd.DataFrame, predictions: pd.DataFrame, failed_molecules: pd.DataFrame
) -> None:
    """
    Print processing summary statistics.

    Args:
        df: Original input DataFrame
        predictions: DataFrame with successful predictions
        failed_molecules: DataFrame with failed molecules
    """
    failed_count: int = len(failed_molecules)
    print(f"\nProcessing Summary:")
    print(f"Total molecules: {len(df)}")
    print(f"Successfully processed: {len(df) - failed_count}")
    print(f"Failed molecules: {failed_count}")

    if not predictions.empty:
        successful_count: int = len(df) - failed_count
        print(f"Total predictions generated: {len(predictions)}")
        if successful_count > 0:
            print(
                f"Average predictions per successful molecule: {len(predictions) / successful_count:.2f}"
            )


def create_failed_molecule_record(
    molecule_id: int,
    parent_name: str,
    failure_reason: str,
    parent_smiles: str | None = None,
) -> pd.DataFrame:
    """
    Create a failed molecule record as a DataFrame.

    Args:
        molecule_id: Index of the molecule
        parent_name: Name/ID of the parent molecule
        failure_reason: Reason for failure
        parent_smiles: SMILES string of parent molecule (optional)

    Returns:
        DataFrame containing failed molecule information
    """
    return pd.DataFrame(
        data=[
            {
                "molecule_id": molecule_id,
                "parent_name": parent_name,
                "failure_reason": failure_reason,
                "parent_smiles": parent_smiles,
            }
        ]
    )

def get_largest_fragment(smiles: str) -> str:
    """
    Get the largest fragment of a molecule.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        SMILES string of the largest fragment
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"Cannot parse SMILES: {smiles}")
    fragments = GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(fragments) > 1:
        largest_fragment = max(fragments, key=lambda y: y.GetNumAtoms())
        return MolToSmiles(largest_fragment)
    else:
        return smiles


def standardize_molecules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize molecules in the DataFrame and filter out problematic molecules.

    This function performs the following operations:
    1. Standardizes molecules using MolVS
    2. Removes salts (keeps only the largest fragment) with RDKit
    3. Filters out molecules containing atoms other than H, C, N, O, P, S, F, Cl, Br, I

    Args:
        df: DataFrame containing molecules (must have 'ROMol' column)

    Returns:
        Tuple of (standardized_df, failed_molecules_df)
    """

    # Initialize lists to track failed molecules
    failed_molecules = []

    # 1. Standardize molecules with MolVS
    print("Standardizing molecules with MolVS...")
    standardizer = Standardizer()

    # Process each molecule
    for idx, row in df.iterrows():
        mol = row["ROMol"]
        molecule_id = int(idx) if isinstance(idx, (int, float, str)) else 0
        parent_name = str(row.get("ID", f"molecule_{idx}"))

        try:
            # Standardize the molecule
            mol = standardizer.standardize(mol)

            # 2. Remove salts - get the largest fragment
            fragments = GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if len(fragments) > 1:
                # Keep only the largest fragment
                largest_fragment = max(fragments, key=lambda x: x.GetNumAtoms())
                mol = largest_fragment

            # 3. Check for allowed atoms only (H, C, N, O, P, S, F, Cl, Br, I)
            allowed_atoms = {1, 6, 7, 8, 15, 16, 9, 17, 35, 53}
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() not in allowed_atoms:
                    failed_molecules.append(
                        create_failed_molecule_record(
                            molecule_id=molecule_id,
                            parent_name=parent_name,
                            failure_reason=f"disallowed_atom_detected: {atom.GetSymbol()}",
                            parent_smiles=MolToSmiles(mol) if mol is not None else None,
                        )
                    )
                    break
            else:
                df.at[idx, "ROMol"] = mol
                continue

        except Exception as e:
            failed_molecules.append(
                create_failed_molecule_record(
                    molecule_id=molecule_id,
                    parent_name=parent_name,
                    failure_reason=f"standardization_error: {str(e)}",
                    parent_smiles=MolToSmiles(mol) if mol is not None else None,
                )
            )
            continue

    # Remove failed molecules from the DataFrame
    failed_indices = [record.iloc[0]["molecule_id"] for record in failed_molecules]
    df_standardized = df.drop(failed_indices).reset_index(drop=True)

    # Combine all failed molecule records
    failed_df = (
        pd.concat(failed_molecules, ignore_index=True)
        if failed_molecules
        else pd.DataFrame()
    )

    print(
        f"Standardization complete: {len(df_standardized)} molecules passed, {len(failed_df)} failed"
    )

    return df_standardized, failed_df
