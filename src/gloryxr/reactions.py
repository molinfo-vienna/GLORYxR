"""
Core chemical reaction processing engine for GLORYxR.

This module handles both abstract reaction management and concrete reaction generation.
"""

import csv
import importlib.resources
import itertools
from typing import Literal, Self

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.rdmolops import AddHs, CombineMols, GetMolFrags, RemoveHs, SanitizeMol
from rdkit.Chem.RegistrationHash import GetStereoTautomerHash
from rdkit.Geometry import Point2D
from rdkit.rdBase import BlockLogs

from gloryxr.som import annotate_educt_and_product_inplace
from gloryxr.utils import MetabolismReaction

__all__ = ["Reactor"]

_rules_data = importlib.resources.files("gloryxr").joinpath("rules_data")


class Reactor:
    """
    Main class for applying reactions with GLORYxR.
    """

    def __init__(
        self,
        reactions: list[ChemicalReaction],
        strict_soms: bool = False,
    ) -> None:
        self.strict_soms: bool = strict_soms
        self.abstract_reactions: list[ChemicalReaction] = [
            _fixup_reaction(rxn) for rxn in reactions
        ]

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
        assert phase in ["1", "2", "1+2"]

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

    def react_one(self, mol: Mol) -> list[MetabolismReaction]:
        """Generate metabolism reactions for a single molecule.

        Args:
            mol: Molecule to perform metabolism reactions on
        """
        concrete_reactions: list[ChemicalReaction] = list(
            itertools.chain.from_iterable(
                (
                    _to_concrete_reactions(
                        reaction=abstract_reaction, educt=mol, pretty=True
                    )
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

        separated_reactions = itertools.chain.from_iterable(
            _separate_reactions_for_products(concrete_reaction)
            for concrete_reaction in concrete_reactions
        )

        # Filter out products with less than 3 heavy atoms
        return [
            MetabolismReaction(rxn)
            for rxn in separated_reactions
            if rxn.GetProductTemplate(0).GetNumHeavyAtoms() >= 3
        ]


def _fixup_reaction(reaction: ChemicalReaction) -> ChemicalReaction:
    fixed = ChemicalReaction()

    for mol in reaction.GetReactants():
        fixed.AddReactantTemplate(mol)
    for mol in reaction.GetAgents():
        fixed.AddAgentTemplate(mol)

    # INFO: This ensures that products are placed in the same Mol
    # object, which seems to be extremely important for the correct
    # application of ring-breaking reactions. Without it, sometimes
    # atoms are randomly deleted.
    combined = None
    for mol in reaction.GetProducts():
        combined = mol if combined is None else CombineMols(combined, mol)
    if combined is not None:
        fixed.AddProductTemplate(combined)

    for name, value in reaction.GetPropsAsDict(
        includePrivate=True, autoConvertStrings=False
    ).items():
        fixed.SetProp(name, value)

    fixed.Initialize()
    return fixed


def _to_concrete_reactions(
    reaction: ChemicalReaction, educt: Mol, pretty: bool
) -> list[ChemicalReaction]:
    # INFO: AddHs is very important for correct application of reactions
    products = itertools.chain.from_iterable(reaction.RunReactants([AddHs(educt)]))

    known_products = set()
    reactions = []

    for product in products:
        if (key := GetStereoTautomerHash(product)) not in known_products:
            known_products.add(key)
        else:
            continue

        try:
            block = BlockLogs()
            SanitizeMol(product)
            product = RemoveHs(product)
            del block
        except Exception:
            continue

        concrete_reaction = ChemicalReaction()
        concrete_reaction.AddReactantTemplate(Mol(educt))
        concrete_reaction.AddProductTemplate(product)

        if pretty:
            _prettify_reaction_products(concrete_reaction)

        for name, value in reaction.GetPropsAsDict(
            includePrivate=True, autoConvertStrings=False
        ).items():
            concrete_reaction.SetProp(name, value)

        reactions.append(concrete_reaction)

    return reactions


def _prettify_reaction_products(rxn: ChemicalReaction):
    def point_to_2d(p):
        return Point2D(p.x, p.y)

    product = rxn.GetProductTemplate(0)
    try:
        conf = product.GetConformer()
    except ValueError:
        return
    coord_map = {
        atom.GetIdx(): point_to_2d(conf.GetAtomPosition(atom.GetIdx()))
        for atom in product.GetAtoms()
        if atom.HasProp("react_atom_idx")
    }
    product.RemoveAllConformers()
    Compute2DCoords(product, coordMap=coord_map)


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
            includePrivate=True, autoConvertStrings=False
        ).items():
            split_reaction.SetProp(name, value)

        results.append(split_reaction)

    return results
