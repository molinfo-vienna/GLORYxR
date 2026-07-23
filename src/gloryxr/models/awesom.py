from os import PathLike
from pathlib import Path
from typing import override

import numpy as np
from awesom.dataset import mol_to_data
from awesom.model import SOMPredictor, predict_ensemble
from rdkit.Chem.rdChemReactions import ChemicalReaction
from torch_geometric.data import DataLoader

from gloryxr.models import ModelProvider

__all__ = ["AweSOMModelProvider"]


class AweSOMModelProvider(ModelProvider):
    def __init__(self, model_path: PathLike[str] | str) -> None:
        self.models = []

        for model_path in sorted(Path(model_path).glob("model_*")):
            checkpoint_path = model_path / "checkpoints" / "best_model.ckpt"
            self.models.append(SOMPredictor.load(str(checkpoint_path)))

    @override
    def predict_proba(
        self,
        reactions: list[ChemicalReaction],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        mol_data = [
            mol_to_data(
                rxn.GetReactantTemplate(0), soms=[], mol_id=mol_id, description=""
            )
            for mol_id, rxn in enumerate(reactions)
        ]

        # NOTE: this will predict scores for all atoms, and then
        # filter. Only predicting atoms marked as SOM by the reaction
        # mechanism would be more efficient.
        predictions = predict_ensemble(DataLoader(mol_data), self.models)

        probabilities_mapped = {
            (mol_id, atom_id): prob
            for mol_id, atom_id, prob in zip(
                predictions.mol_ids.tolist(),
                predictions.atom_ids.tolist(),
                predictions.get_probabilities().mean(dim=0).tolist(),
                strict=True,
            )
        }

        results = []
        for mol_id, rxn in enumerate(reactions):
            results.append(
                max(
                    [
                        probabilities_mapped[(mol_id, atom.GetIdx())]
                        for atom in rxn.GetReactantTemplate(0).GetAtoms()
                        if atom.GetAtomMapNum() != 0
                    ],
                    default=np.nan,
                )
            )

        return np.asarray(results)
