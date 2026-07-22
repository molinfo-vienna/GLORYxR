"""
Core chemical reaction processing engine for GLORYxR.

This module handles both abstract reaction management and concrete reaction generation.
"""

import csv
import importlib.resources
import itertools
from typing import Literal, Self

from rdkit.Chem.inchi import MolToInchi
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts
from rdkit.Chem.rdmolops import AddHs, GetMolFrags, RemoveHs, SanitizeMol
from rdkit.rdBase import BlockLogs

from gloryxr.som import annotate_educt_and_product_inplace

__all__ = ["Reactor"]

_rules_data = importlib.resources.files("gloryxr").joinpath("rules_data")


class Reactor:
    """
    Main class for applying reactions with GLORYxR.
    """

    def __init__(
        self, reactions: list[ChemicalReaction], strict_soms: bool = False
    ) -> None:
        self.strict_soms: bool = strict_soms
        self.abstract_reactions: list[ChemicalReaction] = reactions

    @classmethod
    def load_builtin(
        cls, phase: Literal["1", "2", "1+2"], strict_soms: bool = False
    ) -> Self:
        """
        Load a reactor with one of the built-in sets of metabolism reactions.

        Args:
           phase: {"1", "2", "1+2"} Phase-subset of reactions that should be loaded
           strict_soms: Whether to perform stricter SOM tagging
        """
        abstract_reactions = []

        with _rules_data.joinpath("gloryx_reactionrules_connect.csv").open() as f:
            for row in csv.DictReader(f):
                if phase == "1" and "phase 1" not in row["Name of rule subset"].lower():
                    continue
                if phase == "2" and "phase 2" not in row["Name of rule subset"].lower():
                    continue

                reaction: ChemicalReaction = ReactionFromSmarts(row["SMIRKS"])
                reaction.SetProp("_Name", row["Reaction name"])
                reaction.SetProp("_Priority", row["Priority level"])
                reaction.SetProp("_Subset", row["Name of rule subset"])

                abstract_reactions.append(reaction)

        return cls(abstract_reactions, strict_soms)

    def react_one(self, mol: Mol) -> list[ChemicalReaction]:
        concrete_reactions: list[ChemicalReaction] = list(
            itertools.chain.from_iterable(
                (
                    _to_concrete_reactions(reaction=abstract_reaction, educt=mol)
                    for abstract_reaction in self.abstract_reactions
                )
            )
        )

        # Annotate each concrete reaction with SOM information
        for concrete_reaction in concrete_reactions:
            annotate_educt_and_product_inplace(
                educt=concrete_reaction.GetReactants()[0],
                product=concrete_reaction.GetProducts()[0],
                strict_soms=self.strict_soms,
            )

        return list(
            itertools.chain.from_iterable(
                _separate_reactions_for_products(concrete_reaction)
                for concrete_reaction in concrete_reactions
            )
        )


def _to_concrete_reactions(
    reaction: ChemicalReaction, educt: Mol
) -> list[ChemicalReaction]:
    # INFO: AddHs is very important for correct application of reactions
    products = itertools.chain.from_iterable(reaction.RunReactants([AddHs(educt)]))

    known_products = set()
    reactions = []

    for product in products:
        try:
            block = BlockLogs()
            SanitizeMol(product)
            product = RemoveHs(product)
            del block
        except Exception:
            continue

        # Check for duplicate products using InChI
        if (inchi := MolToInchi(product)) not in known_products:
            known_products.add(inchi)
        else:
            continue

        concrete_reaction = ChemicalReaction()
        concrete_reaction.AddReactantTemplate(Mol(educt))
        concrete_reaction.AddProductTemplate(product)

        for name, value in reaction.GetPropsAsDict(includePrivate=True).items():
            concrete_reaction.SetProp(name, value)

        reactions.append(concrete_reaction)

    return reactions


def _separate_reactions_for_products(
    combined_reaction: ChemicalReaction,
) -> list[ChemicalReaction]:
    product_or_products = combined_reaction.GetProducts()[0]
    products = GetMolFrags(product_or_products, asMols=True, sanitizeFrags=False)

    if len(products) == 1:
        return [combined_reaction]

    results = []
    for product in products:
        split_reaction = ChemicalReaction()
        split_reaction.AddReactantTemplate(combined_reaction.GetReactants()[0])
        split_reaction.AddProductTemplate(product)

        for name, value in combined_reaction.GetPropsAsDict(
            includePrivate=True
        ).items():
            split_reaction.SetProp(name, value)

        results.append(split_reaction)

    return results
