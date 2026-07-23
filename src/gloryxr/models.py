from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, Literal, override

import joblib
import numpy as np
from awesom.dataset import mol_to_data
from awesom.model import SOMPredictor, predict_ensemble
from fame3r import FAME3RVectorizer
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdmolfiles import MolToSmiles
from torch_geometric.loader import DataLoader

from gloryxr.utils import extract_smiles_for_soms

__all__ = ["ModelProvider", "LocalFAME3RModelProvider"]


class ModelProvider(ABC):
    """Base class for GLORYxR model providers"""

    @abstractmethod
    def predict_proba(
        self,
        reactions: list[ChemicalReaction],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Predict probabilities for a list of metabolism reactions, tagged with SOMs."""
        ...


class MultiFAME3RModelProvider(ModelProvider):
    """A model provider for a family of reaction-class specific FAME3R models.

    This when used with our proivided FAME3R models, predictions
    generated using this class will closely follow those provided by the
    original GLORYx implementation.
    """

    def __init__(self, model_path: PathLike[str] | str) -> None:
        self.models: dict[str, Any] = {}
        self.vectorizer = FAME3RVectorizer().fit()

        model_paths = list(Path(model_path).glob("*.joblib"))
        if len(model_paths) == 0:
            raise RuntimeError(f"No models could be found at '{model_path}'")

        for model_path in model_paths:
            self.models[model_path.stem] = joblib.load(filename=model_path)

    @override
    def predict_proba(
        self,
        reactions: list[ChemicalReaction],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        results = []

        for rxn in reactions:
            priority = rxn.GetProp("_Priority")
            if priority == "uncommon":
                priority_factor = 0.2
            else:
                priority_factor = 1.0
            subset = rxn.GetProp("_Subset")

            reactive_atoms = extract_smiles_for_soms(rxn.GetReactantTemplate(0))
            if len(reactive_atoms) == 0:
                # TODO: fix this using rules
                results.append(np.nan)
                continue

            descriptors = self.vectorizer.transform(
                np.array(reactive_atoms).reshape(-1, 1)
            )
            predictions = self.models[subset].predict_proba(descriptors)

            results.append(predictions[:, -1].max() * priority_factor)

        return np.asarray(results)


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
            mol_to_data(rxn.GetReactantTemplate(0), soms=[], mol_id=mol_id, description="")
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
