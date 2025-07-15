# pyright: basic

"""
Functions to convert abstract reactions to concrete reactions.
"""

import itertools

from rdkit.Chem.inchi import MolToInchi
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdmolops import AddHs, RemoveHs, SanitizeMol
from rdkit.rdBase import BlockLogs


# TODO: fix what happens when reaction matches multiple times!
def to_concrete_reactions(
    reaction: ChemicalReaction, educt: Mol
) -> list[ChemicalReaction]:
    products = itertools.chain.from_iterable(reaction.RunReactants([AddHs(educt)]))

    known_products = set()
    reactions = []

    for product in products:
        try:
            block = BlockLogs()
            SanitizeMol(product)
            del block
        except Exception:
            continue

        if (inchi := MolToInchi(product)) not in known_products:
            known_products.add(inchi)
        else:
            continue

        product_ = AddHs(product)
        educt_ = Mol(educt)

        concrete_reaction = ChemicalReaction()
        concrete_reaction.AddReactantTemplate(RemoveHs(educt_))
        concrete_reaction.AddProductTemplate(RemoveHs(product_))
        if reaction.HasProp("_Name"):
            concrete_reaction.SetProp("_Name", reaction.GetProp("_Name"))

        reactions.append(concrete_reaction)

    return reactions
