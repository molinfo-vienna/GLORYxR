#pyright: basic

"""
Core chemical reaction processing engine for GLORYxR.
"""

import csv
import importlib.resources
import itertools

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts

from gloryxr.reaction import to_concrete_reactions
from gloryxr.som import annotate_educt_and_product_inplace


rules_data = importlib.resources.files("gloryxr").joinpath("rules_data")


class Reactor:
    def __init__(self, strict_soms: bool = False) -> None:
        """
        Initialize the Reactor.

        Args:
            strict_soms: Whether to use strict SOM validation
        """
        self.strict_soms: bool = strict_soms
        self.abstract_reactions: list[ChemicalReaction] = []

        with rules_data.joinpath("gloryx_reactionrules.csv").open() as f:
            for row in csv.DictReader(f):
                reaction: ChemicalReaction = ReactionFromSmarts(row["SMIRKS"])
                reaction.SetProp("_Name", row["Reaction name"])
                reaction.SetProp("_Priority", row["Priority level"])
                reaction.SetProp("_Subset", row["Name of rule subset"])

                self.abstract_reactions.append(reaction)

    def react_one(self, mol: Mol) -> list[ChemicalReaction]:
        """
        Applies abstract reactions to a molecule to generate concrete reactions.

        Args:
            mol: RDKit molecule object

        Returns:
            List of concrete reactions
        """
        concrete_reactions: list[ChemicalReaction] = list(
            itertools.chain.from_iterable(
                (
                    to_concrete_reactions(reaction=abstract_reaction, educt=mol)
                    for abstract_reaction in self.abstract_reactions
                )
            )
        )

        for concrete_reaction in concrete_reactions:
            annotate_educt_and_product_inplace(
                educt=concrete_reaction.GetReactants()[0],
                product=concrete_reaction.GetProducts()[0],
                strict_soms=self.strict_soms,
            )

        return concrete_reactions