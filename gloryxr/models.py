from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, Literal, override

import joblib
import numpy as np
from fame3r import FAME3RVectorizer
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdmolfiles import MolToSmiles

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


class LocalFAME3RModelProvider(ModelProvider):
    """A model provider that loads a family of reaction-class specific FAME3R models.

    This when used with our proivided FAME3R models, predictions
    generated using this class will closely follow the behavior of the
    original GLORYx paper.
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
