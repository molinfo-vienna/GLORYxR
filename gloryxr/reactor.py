# pyright: basic

"""
Core chemical reaction processing engine for GLORYxR.

This module handles both abstract reaction management and concrete reaction generation.
"""

import csv
import importlib.resources
import itertools

from rdkit.Chem.inchi import MolToInchi
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts
from rdkit.Chem.rdmolops import AddHs, RemoveHs, SanitizeMol
from rdkit.rdBase import BlockLogs

from gloryxr.som import annotate_educt_and_product_inplace

rules_data = importlib.resources.files("gloryxr").joinpath("rules_data")


class Reactor:
    """
    Core chemical reaction processing engine for GLORYxR.

    This class manages abstract reactions and provides methods to generate
    concrete reactions from input molecules.
    """

    def __init__(self, strict_soms: bool = False) -> None:
        """
        Initialize the Reactor with reaction rules.

        Args:
            strict_soms: Whether to use strict SOM validation
        """
        self.strict_soms: bool = strict_soms
        self.abstract_reactions: list[ChemicalReaction] = []

        # Load reaction rules from CSV file
        self._load_reaction_rules()

    def _load_reaction_rules(self) -> None:
        """Load abstract reaction rules from the CSV file."""
        with rules_data.joinpath("gloryx_reactionrules.csv").open() as f:
            for row in csv.DictReader(f):
                reaction: ChemicalReaction = ReactionFromSmarts(row["SMIRKS"])
                reaction.SetProp("_Name", row["Reaction name"])
                reaction.SetProp("_Priority", row["Priority level"])
                reaction.SetProp("_Subset", row["Name of rule subset"])

                self.abstract_reactions.append(reaction)

    def _to_concrete_reactions(
        self, reaction: ChemicalReaction, educt: Mol
    ) -> list[ChemicalReaction]:
        """
        Convert an abstract reaction to concrete reactions for a given educt.

        This method applies the abstract reaction to the educt molecule and
        generates all possible concrete reactions, filtering out duplicates
        and invalid products.

        Args:
            reaction: Abstract chemical reaction
            educt: Input molecule

        Returns:
            List of concrete reactions
        """
        # Generate all possible products from the reaction
        products = itertools.chain.from_iterable(reaction.RunReactants([AddHs(educt)]))

        known_products = set()
        reactions = []

        for product in products:
            # Sanitize the product molecule
            try:
                block = BlockLogs()
                SanitizeMol(product)
                del block
            except Exception:
                # Skip invalid products
                continue

            # Check for duplicate products using InChI
            if (inchi := MolToInchi(product)) not in known_products:
                known_products.add(inchi)
            else:
                continue

            # Create concrete reaction
            product_ = AddHs(product)
            educt_ = Mol(educt)

            concrete_reaction = ChemicalReaction()
            concrete_reaction.AddReactantTemplate(RemoveHs(educt_))
            concrete_reaction.AddProductTemplate(RemoveHs(product_))

            # Copy reaction name, priority, and subset if available
            if reaction.HasProp("_Name"):
                concrete_reaction.SetProp("_Name", reaction.GetProp("_Name"))
            if reaction.HasProp("_Priority"):
                concrete_reaction.SetProp("_Priority", reaction.GetProp("_Priority"))
            if reaction.HasProp("_Subset"):
                concrete_reaction.SetProp("_Subset", reaction.GetProp("_Subset"))

            reactions.append(concrete_reaction)

        return reactions

    def react_one(self, mol: Mol) -> list[ChemicalReaction]:
        """
        Applies abstract reactions to a molecule to generate concrete reactions.

        Args:
            mol: RDKit molecule object

        Returns:
            List of concrete reactions with SOM annotations
        """
        concrete_reactions: list[ChemicalReaction] = list(
            itertools.chain.from_iterable(
                (
                    self._to_concrete_reactions(reaction=abstract_reaction, educt=mol)
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

        return concrete_reactions
