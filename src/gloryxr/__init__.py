"""
Metabolite prediction using GLORYxR.
"""

import itertools

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolToSmiles

from gloryxr.models import ModelProvider
from gloryxr.reactions import Reactor
from gloryxr.utils import MetabolismReaction, mol_without_mappings

__all__ = ["GLORYxR", "Reactor", "ModelProvider"]


class GLORYxR:
    """
    Main class for metabolite prediction using GLORYxR.

    Args:
        strict_soms: Whether to use strict SOMs
    """

    def __init__(
        self,
        *,
        models: ModelProvider,
        reactor: Reactor,
    ) -> None:
        self.model_provider = models
        self.reactor = reactor

    def predict(self, mols: list[Mol]) -> list[MetabolismReaction]:
        """
        Generate metabolism predictions for a list of molecules.

        Args:
            mols: List of molecules to perform metabolism prediction for
        """
        predictions = itertools.chain.from_iterable(
            (self.predict_one(mol) for mol in mols)
        )

        return list(predictions)

    def predict_one(self, mol: Mol) -> list[MetabolismReaction]:
        """
        Generate metabolism predictions for a single molecule.

        Args:
            mol: Molecule to perform metabolism prediction for
        """
        predicted_reactions = [
            MetabolismReaction(concrete_reaction)
            for concrete_reaction in self.reactor.react_one(mol)
        ]
        for reaction in predicted_reactions:
            score = self.model_provider.predict_proba([reaction])[0]
            reaction.SetDoubleProp("Score", score)

        # Deduplicate predicted products (keeping the one with highest score)
        deduplicated: dict[str, MetabolismReaction] = {}
        for reaction in predicted_reactions:
            product_smiles = MolToSmiles(
                # TODO: This gives the expected results, but is hard
                # to understand and possibly wrong...
                mol_without_mappings(reaction.GetProductTemplate(0))
            )
            if product_smiles not in deduplicated or deduplicated[
                product_smiles
            ].GetDoubleProp("Score") < reaction.GetDoubleProp("Score"):
                deduplicated[product_smiles] = reaction

        return list(deduplicated.values())
