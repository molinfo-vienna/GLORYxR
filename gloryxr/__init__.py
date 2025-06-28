# pyright: basic

import csv
import importlib.resources
import itertools

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts

from gloryxr import util
from gloryxr.reaction import to_concrete_reactions
from gloryxr.som import annotate_educt_and_product_inplace

__all__ = ["Reactor", "util"]

rules_data = importlib.resources.files("gloryxr").joinpath("rules_data")


class Reactor:
    def __init__(self, strict_soms: bool = False) -> None:
        self.strict_soms = strict_soms
        self.reactions: list[ChemicalReaction] = []

        with rules_data.joinpath("gloryx_reactionrules.csv").open() as f:
            for row in csv.DictReader(f):
                reaction = ReactionFromSmarts(row["SMIRKS"])
                reaction.SetProp("_Name", row["Reaction name"])

                self.reactions.append(reaction)

    def react_one(self, mol: Mol) -> list[ChemicalReaction]:
        concrete_reactions = list(
            itertools.chain.from_iterable(
                (to_concrete_reactions(reaction, mol) for reaction in self.reactions)
            )
        )

        for concrete_reaction in concrete_reactions:
            annotate_educt_and_product_inplace(
                concrete_reaction.GetReactants()[0],
                concrete_reaction.GetProducts()[0],
                strict_soms=self.strict_soms,
            )

        return concrete_reactions
