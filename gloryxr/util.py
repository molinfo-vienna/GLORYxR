from collections.abc import Iterable

import pandas as pd
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdchem import Mol


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


def extract_similes_for_soms(mol: Mol) -> list[Mol]:
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

    return results
