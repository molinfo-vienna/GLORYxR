import json

import pandas as pd
from rdkit.Chem import PandasTools
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles


def process_metabolites(df) -> pd.DataFrame:
    """
    Process the metabolites column to create a more structured format.
    Handles the nested metabolite structure in test data.
    Explodes metabolites into separate rows for easier analysis.

    Args:
        df (pd.DataFrame): DataFrame with metabolites column

    Returns:
        pd.DataFrame: DataFrame with processed metabolites (one row per metabolite)
    """
    # Explode metabolites into separate rows
    metabolites_exploded: pd.DataFrame = df.explode("metabolites")

    # Normalize metabolites data into separate columns
    metabolites_df: pd.DataFrame = pd.json_normalize(
        data=metabolites_exploded["metabolites"].tolist()
    )

    # Rename metabolite columns to match reference data structure
    metabolites_df.rename(
        columns={
            "smiles": "metabolite_smiles",
            "generation": "generation",
            "metaboliteName": "metabolite_name",
        },
        inplace=True,
    )

    metabolites_df["metabolite_smiles"] = metabolites_df["metabolite_smiles"].apply(
        lambda x: MolToSmiles(
            MolFromSmiles(x), canonical=True, ignoreAtomMapNumbers=True
        )
    )

    # Combine parent data with metabolite data
    result_df: pd.DataFrame = pd.concat(
        [
            metabolites_exploded.drop("metabolites", axis=1).reset_index(drop=True),
            metabolites_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return result_df


def main():
    """Main function to process test dataset."""
    print("Processing test dataset...")

    # Load and parse test data
    with open(
        file="GLORYx_data/test/gloryx_test_dataset.json", mode="r", encoding="utf-8"
    ) as f:
        test_data = json.load(f)

    # Convert to DataFrame and rename columns for consistency
    test_df = pd.DataFrame(test_data)
    test_df = test_df.rename(
        columns={"drugName": "parent_name", "smiles": "parent_smiles"}
    )

    # Canonicalize SMILES
    test_df["parent_smiles"] = test_df["parent_smiles"].apply(
        lambda x: MolToSmiles(
            MolFromSmiles(x), canonical=True, ignoreAtomMapNumbers=True
        )
    )

    # Add RDKit mol objects for parent molecules
    test_df["parent_mol"] = test_df["parent_smiles"].apply(MolFromSmiles)

    # Filter out invalid SMILES
    problematic_smiles = test_df[test_df["parent_mol"].isna()]["parent_smiles"].tolist()
    if problematic_smiles:
        print(
            f"Warning: Found {len(problematic_smiles)} problematic SMILES: {problematic_smiles}"
        )
    test_df = test_df[test_df["parent_mol"].notna()]

    # Save parent molecules to SDF
    PandasTools.WriteSDF(
        df=test_df,
        out="GLORYx_data/test/gloryx_test_dataset.sdf",
        molColName="parent_mol",
        idName="parent_name",
        properties=["doi", "parent_smiles"],
    )
    print(f"Saved test dataset to SDF: {len(test_df)} parent molecules")

    # Process metabolites
    print("Processing metabolites...")
    test_metabolites_df = process_metabolites(test_df)

    # Save metabolites to CSV with specific columns
    metabolite_csv_columns = [
        "parent_name",
        "parent_smiles",
        "metabolite_name",
        "metabolite_smiles",
        "generation",
        "doi",
    ]
    test_metabolites_df.to_csv(
        "GLORYx_data/test/gloryx_test_dataset_metabolites_exploded.csv",
        columns=metabolite_csv_columns,
        header=True,
        index=False,
    )
    print(f"Saved metabolites to CSV: {len(test_metabolites_df)} metabolite entries")

    # Add RDKit mol objects for metabolites
    test_metabolites_df["metabolite_mol"] = test_metabolites_df[
        "metabolite_smiles"
    ].apply(MolFromSmiles)

    # Filter out invalid metabolite SMILES
    problematic_metabolite_smiles = test_metabolites_df[
        test_metabolites_df["metabolite_mol"].isna()
    ]["metabolite_smiles"].tolist()
    if problematic_metabolite_smiles:
        print(
            f"Warning: Found {len(problematic_metabolite_smiles)} problematic metabolite SMILES: {problematic_metabolite_smiles}"
        )
    test_metabolites_df = test_metabolites_df[
        test_metabolites_df["metabolite_mol"].notna()
    ]

    # Save metabolites to SDF
    PandasTools.WriteSDF(
        df=test_metabolites_df,
        out="GLORYx_data/test/gloryx_test_dataset_metabolites_exploded.sdf",
        molColName="metabolite_mol",
        idName="metabolite_name",
        properties=[
            "parent_name",
            "parent_smiles",
            "doi",
            "metabolite_smiles",
            "generation",
        ],
    )
    print(f"Saved metabolites to SDF: {len(test_metabolites_df)} metabolite entries")

    return test_df, test_metabolites_df


if __name__ == "__main__":
    test_df, test_metabolites_df = main()
