"""
Metabolite prediction using GLORYxR.
"""

import itertools
from dataclasses import dataclass
from typing import Literal

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdmolfiles import MolToSmiles

from gloryxr._models import _LocalModelProvider, _ModelProvider
from gloryxr.reactions import Reactor
from gloryxr.should_be_in_fame3r import Fame3RVectorizer
from gloryxr.utils import (
    extract_smiles_for_soms,
    mol_without_mappings,
)

__all__ = ["GLORYxR", "Prediction"]


class GLORYxR:
    """
    Main class for metabolite prediction using GLORYxR.

    Args:
        strict_soms: Whether to use strict SOMs
    """

    def __init__(
        self,
        *,
        strict_soms: bool = False,
        _models: type[_ModelProvider] = _LocalModelProvider,
    ) -> None:
        self.model_provider = _models()
        self.vectorizer = Fame3RVectorizer().fit()
        self.reactor = Reactor(strict_soms=strict_soms)

    def predict(self, mols: list[Mol]) -> list["Prediction"]:
        """
        Generate metabolism predictions for a list of molecules.

        Args:
            mols: List of molecules to perform metabolism prediction for
        """
        predictions = itertools.chain.from_iterable(
            (self.predict_one(mol) for mol in mols)
        )

        return list(predictions)

    def predict_one(self, mol: Mol) -> list["Prediction"]:
        """
        Generate metabolism predictions for a single molecule.

        Args:
            mol: Molecule to perform metabolism prediction for
        """
        predictions = [
            Prediction(
                concrete_reaction=concrete_reaction,
                score=self._generate_predictions(
                    concrete_reaction.GetReactants()[0],
                    concrete_reaction.GetProp("_Priority"),
                    concrete_reaction.GetProp("_Subset"),
                ),
            )
            for concrete_reaction in self.reactor.react_one(mol)
        ]

        # Deduplicate predicted products
        deduplicated: dict[str, Prediction] = {}
        for prediction in predictions:
            product_smiles = prediction.get_product_smiles()
            if (
                product_smiles not in deduplicated
                or deduplicated[product_smiles].score < prediction.score
            ):
                deduplicated[product_smiles] = prediction
        predictions = list(deduplicated.values())

        # Filter out products with less than 3 heavy atoms
        predictions = [
            pred for pred in predictions if pred.product.GetNumHeavyAtoms() >= 3
        ]

        return list(predictions)

    def _generate_predictions(
        self,
        marked_educt: Mol,
        priority: Literal["common", "uncommon"],
        subset: str,
    ) -> float:
        som_smiles = extract_smiles_for_soms(marked_educt)
        descriptors = [
            self.vectorizer.transform_one(som_smile) for som_smile in som_smiles
        ]
        scores = [self._get_prediction_score(d, priority, subset) for d in descriptors]

        return max(scores) if scores else float("nan")

    def _get_prediction_score(
        self, descriptors, priority: Literal["common", "uncommon"], subset: str
    ) -> float:
        if priority == "common":
            priority_factor = 1.0
        elif priority == "uncommon":
            priority_factor = 0.2
        else:
            raise ValueError(f"Invalid priority: {priority}")

        som_probability = self.model_provider.predict_proba(
            subset=subset, descriptors=[descriptors]
        )[0][-1]

        return som_probability * priority_factor


@dataclass
class Prediction:
    """
    Class that encapsulates a single reaction prediction.

    Args:
        concrete_reaction: The specific reaction that was predicted.
        score: The probability score of the predicted reaction, relative to other reactions.
    """

    concrete_reaction: ChemicalReaction
    score: float

    @property
    def educt(self) -> Mol:
        """Educt molecule of the predicted reaction."""
        return self.concrete_reaction.GetReactants()[0]

    @property
    def product(self) -> Mol:
        """Product molecule of the predicted reaction."""
        return self.concrete_reaction.GetProducts()[0]

    def get_educt_smiles(self, clean: bool = True) -> str:
        """
        Generate SMILES string for the educt of the predicted reaction.

        Args:
           clean: Whether to remove mapping information from the returned SMILES
        """
        mol = mol_without_mappings(self.educt) if clean else self.educt
        return MolToSmiles(mol, ignoreAtomMapNumbers=True)

    def get_product_smiles(self, clean: bool = True) -> str:
        """
        Generate SMILES string for the product of the predicted reaction.

        Args:
           clean: Whether to remove mapping information from the returned SMILES
        """
        mol = mol_without_mappings(self.product) if clean else self.product
        return MolToSmiles(mol, ignoreAtomMapNumbers=True)
