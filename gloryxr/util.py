from collections.abc import Iterable

import pandas as pd
from rdkit.Chem.rdChemReactions import ChemicalReaction


def reactions_to_table(reactions: Iterable[ChemicalReaction]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Educt": reaction.GetReactants()[0],
                "Product": reaction.GetProducts()[0],
                "Reaction": reaction.GetProp("_Name")
                if reaction.HasProp("_Name")
                else None,
            }
            for reaction in reactions
        ]
    ).rename_axis("ID")
