"""
Utility functions and classes for GLORYxR metabolite prediction.
"""

from rdkit.Chem import Draw
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdmolfiles import MolToSmiles

__all__ = ["MetabolismReaction", "extract_smiles_for_soms", "mol_without_mappings"]


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


class MetabolismReaction(ChemicalReaction):
    """Transparent wrapper class around :class:`~rdkit.Chem.rdChemReactions.ChemicalReaction`.

    This class can be used in place of a plain
    :class:`~rdkit.Chem.rdChemReactions.ChemicalReaction`, but provides
    better display handling in Jupyter sessions and similar
    environments.

    """

    def _repr_svg_(self):
        return Draw.ReactionToImage(
            self,
            useSVG=True,
            highlightByReactant=True,
            highlightColorsReactants=[(1, 0.502, 0.502)],
        )

    def _repr_png_(self):
        return Draw.ReactionToImage(
            self,
            returnPNG=True,
            highlightByReactant=True,
            highlightColorsReactants=[(1, 0.502, 0.502)],
        )

    def _repr_html_(self):
        prop_rows = [
            f'<tr><th>{key}</th><td style="text-align: left">{value}</td></tr>'
            for key, value in self.GetPropsAsDict(includePrivate=True).items()
        ]

        return (
            "<div><table><tbody><tr><td colspan=2>"
            + self._repr_svg_()
            + "</td></tr>"
            + "".join(prop_rows)
            + "</tbody></table></div>"
        )
