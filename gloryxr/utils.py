# pyright: basic

"""
Utility functions for GLORYxR metabolite prediction.
"""

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdmolfiles import MolToSmiles


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
                "Priority": reaction.GetProp("_Priority")
                if reaction.HasProp("_Priority")
                else None,
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


def mol_without_mappings(mol: Mol) -> Mol:
    """
    Remove atom mapping number information from a molecule.

    Args:
        mol: RDKit molecule object

    Returns:
        Copy of the given molecule with mapping information removed.
    """
    mol_ = Mol(mol)
    for atom in mol_.GetAtoms():
        atom.SetAtomMapNum(0)

    return mol_
