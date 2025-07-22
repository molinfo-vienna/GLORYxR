"""
Functions to annotate the educt molecule with SOM indices.
"""

import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import GetDistanceMatrix


def annotate_educt_and_product_inplace(
    educt: Mol, product: Mol, strict_soms: bool = False
) -> None:
    """
    Annotate the educt and product molecules with SOM indices.
    """

    product_idxs = (
        _get_strict_som_indices(educt, product)
        if strict_soms
        else _get_loose_som_indices(product)
    )
    for idx in product_idxs:
        atom = product.GetAtomWithIdx(idx)
        mapno = atom.GetIntProp("old_mapno") if atom.HasProp("old_mapno") else 1
        atom.SetAtomMapNum(mapno)
        educt.GetAtomWithIdx(atom.GetIntProp("react_atom_idx")).SetAtomMapNum(mapno)


def _get_loose_som_indices(product: Mol) -> list[int]:
    return [
        atom.GetIdx()
        for atom in product.GetAtoms()
        if atom.HasProp("old_mapno") and atom.GetAtomicNum() != 1
    ]


def _get_strict_som_indices(educt: Mol, product: Mol) -> list[int]:
    involved_idx_mappings = {
        atom.GetIntProp("react_atom_idx"): atom.GetIdx()
        for atom in product.GetAtoms()
        if atom.HasProp("react_atom_idx") and atom.GetAtomicNum() != 1
    }

    added_by_reaction_idx = [
        atom.GetIdx()
        for atom in product.GetAtoms()
        if not atom.HasProp("react_atom_idx")
    ]
    removed_by_reaction_idx = [
        atom.GetIdx()
        for atom in educt.GetAtoms()
        if atom.GetIdx() not in involved_idx_mappings
    ]

    if len(removed_by_reaction_idx) != 0:
        return [
            involved_idx_mappings[idx]
            for idx in _get_closest_idxs(
                educt, removed_by_reaction_idx, list(involved_idx_mappings.keys())
            )
        ]
    elif len(added_by_reaction_idx) != 0:
        return _get_closest_idxs(
            product, added_by_reaction_idx, list(involved_idx_mappings.values())
        )
    else:
        return []


def _get_closest_idxs(
    mol: Mol, reference_idx_: list[int], filter_idx_: list[int]
) -> list[int]:
    reference_idx = np.asarray(reference_idx_, dtype=int)
    filter_idx = np.asarray(filter_idx_, dtype=int)

    distances = GetDistanceMatrix(mol)[:, reference_idx].min(axis=1)[filter_idx]
    closest_indices = filter_idx[
        np.argwhere(distances == distances.min(initial=np.inf)).flatten()
    ]

    return [int(idx) for idx in closest_indices]
