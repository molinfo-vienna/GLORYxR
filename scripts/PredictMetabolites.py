from datetime import datetime
import os

from rdkit.Chem import MolToSmiles, PandasTools

from gloryxr import GLORYxR
from gloryxr.utils import reactions_to_table

ROOT = os.getcwd()

DATA_PATH = os.path.join(ROOT, "datasets/GLORYx_data/test/gloryx_test_dataset.sdf")
OUTPUT_DIR = os.path.join(ROOT, "datasets/GLORYx_data/test/results_loose_priority/phase_1_2")

if __name__ == "__main__":
    start_time = datetime.now()

    # Initialize GLORYxR predictor
    gloryxr = GLORYxR(phase=3, strict_soms=False)

    # Import data
    data = PandasTools.LoadSDF(DATA_PATH, removeHs=False)

    # Predict metabolites
    predictions = gloryxr.predict(
        data["ROMol"].to_list()
    )

    # Save results to a Pandas DataFrame
    df = reactions_to_table([rxn.concrete_reaction for rxn in predictions]).assign(
        Score=[rxn.score for rxn in predictions]
    )

    df["parent_smiles"] = df["Educt"].apply(lambda mol: MolToSmiles(mol))
    df["metabolite_smiles"] = df["Product"].apply(lambda mol: MolToSmiles(mol))

    # Rename columns for better compatibility with NERDD output
    df = df.rename(
        columns={
            "Educt": "parent_molecule",
            "Product": "metabolite_molecule",
            "Reaction": "reaction_type",
            "Subset": "reaction_subset",
            "Priority": "priority",
            "Score": "score",
        }
    )

    # Save results to a CSV file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)

    end_time = datetime.now()
    print(f"Prediction completed in: {end_time - start_time}")
