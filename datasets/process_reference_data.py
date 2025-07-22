import json

import pandas as pd
from rdkit.Chem import PandasTools
from rdkit.Chem.rdmolfiles import MolFromSmiles


def process_metabolites(df) -> pd.DataFrame:
    """
    Process the Metabolites column to create a more structured format.
    Explodes metabolites into separate rows for easier analysis.

    Args:
        df (pd.DataFrame): DataFrame with Metabolites column

    Returns:
        pd.DataFrame: DataFrame with processed metabolites (one row per metabolite)
    """
    # Explode metabolites into separate rows
    metabolites_exploded: pd.DataFrame = df.explode("Metabolites")

    # Normalize metabolites data into separate columns
    metabolites_df: pd.DataFrame = pd.json_normalize(
        data=metabolites_exploded["Metabolites"].tolist()
    )

    # Rename metabolite columns for clarity
    metabolites_df.rename(
        columns={
            "SMILES": "metabolite_smiles",
            "InChI": "metabolite_inchi",
            "Phase": "metabolite_phase",
            "DrugBank ID": "metabolite_drugbank_id",
            "MetXBioDB biotransformation ID": "metabolite_metxbiodb_biotransformation_id",
        },
        inplace=True,
    )

    # Combine parent data with metabolite data
    result_df: pd.DataFrame = pd.concat(
        [
            metabolites_exploded.drop("Metabolites", axis=1).reset_index(drop=True),
            metabolites_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return result_df


def load_and_process_reference_data(json_file_path):
    """
    Load JSON data and explode Parent molecule columns into separate columns.

    Args:
        json_file_path (str): Path to the JSON file containing the data

    Returns:
        pd.DataFrame: DataFrame with exploded Parent molecule columns
    """
    # Read and parse JSON data
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Explode Parent molecule columns into separate columns
    parent_mol_df = pd.json_normalize(df["Parent molecule"].tolist())

    # Rename columns to be more descriptive
    parent_mol_df = parent_mol_df.rename(
        columns={
            "SMILES": "parent_smiles",
            "InChI": "parent_inchi",
            "DrugBank ID": "parent_drugbank_id",
            "MetXBioDB ID": "parent_metxbiodb_id",
        }
    )

    # Combine parent molecule data with original DataFrame
    result_df = pd.concat([parent_mol_df, df.drop("Parent molecule", axis=1)], axis=1)

    return result_df


def main():
    """Main function to process reference dataset."""
    print("Processing reference dataset...")

    # Load and process reference data
    reference_df = load_and_process_reference_data(
        "GLORYx_data/reference/gloryx_reference_dataset.json"
    )

    # Add RDKit mol objects for parent molecules
    reference_df["parent_mol"] = reference_df["parent_smiles"].apply(MolFromSmiles)

    # Filter out invalid SMILES
    problematic_smiles = reference_df[reference_df["parent_mol"].isna()][
        "parent_smiles"
    ].tolist()
    if problematic_smiles:
        print(
            f"Warning: Found {len(problematic_smiles)} problematic SMILES: {problematic_smiles}"
        )
    reference_df = reference_df[reference_df["parent_mol"].notna()]

    # Add index for SDF identification
    reference_df["index"] = reference_df.index

    # Save parent molecules to SDF
    PandasTools.WriteSDF(
        df=reference_df,
        out="GLORYx_data/reference/gloryx_reference_dataset.sdf",
        molColName="parent_mol",
        idName="index",
        properties=[
            "parent_smiles",
            "parent_inchi",
            "parent_drugbank_id",
            "parent_metxbiodb_id",
        ],
    )
    print(f"Saved reference dataset to SDF: {len(reference_df)} parent molecules")

    # Process metabolites
    print("Processing metabolites...")
    reference_metabolites_df = process_metabolites(reference_df)

    # Save metabolites to CSV
    reference_metabolites_df.to_csv(
        "GLORYx_data/reference/gloryx_reference_dataset_metabolites_exploded.csv",
        index=False,
    )
    print(
        f"Saved metabolites to CSV: {len(reference_metabolites_df)} metabolite entries"
    )

    # Add RDKit mol objects for metabolites
    reference_metabolites_df["metabolite_mol"] = reference_metabolites_df[
        "metabolite_smiles"
    ].apply(MolFromSmiles)

    # Filter out invalid metabolite SMILES
    problematic_metabolite_smiles = reference_metabolites_df[
        reference_metabolites_df["metabolite_mol"].isna()
    ]["metabolite_smiles"].tolist()
    if problematic_metabolite_smiles:
        print(
            f"Warning: Found {len(problematic_metabolite_smiles)} problematic metabolite SMILES: {problematic_metabolite_smiles}"
        )
    reference_metabolites_df = reference_metabolites_df[
        reference_metabolites_df["metabolite_mol"].notna()
    ]

    # Save metabolites to SDF
    PandasTools.WriteSDF(
        df=reference_metabolites_df,
        out="GLORYx_data/reference/gloryx_reference_dataset_metabolites_exploded.sdf",
        molColName="metabolite_mol",
        idName="metabolite_metxbiodb_biotransformation_id",
        properties=[
            "parent_smiles",
            "parent_inchi",
            "parent_drugbank_id",
            "parent_metxbiodb_id",
            "metabolite_smiles",
            "metabolite_inchi",
            "metabolite_phase",
            "metabolite_drugbank_id",
        ],
    )
    print(
        f"Saved metabolites to SDF: {len(reference_metabolites_df)} metabolite entries"
    )

    return reference_df, reference_metabolites_df


if __name__ == "__main__":
    reference_df, reference_metabolites_df = main()
