from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, Literal, override

import joblib
import numpy as np
import numpy.typing as npt
from fame3r import FAME3RVectorizer
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdmolfiles import MolToSmiles

from gloryxr.utils import extract_smiles_for_soms


class ModelProvider(ABC):
    @abstractmethod
    def predict_proba(
        self,
        reactions: list[ChemicalReaction],
    ) -> npt.NDArray[np.float64]: ...


class LocalFAME3RModelProvider(ModelProvider):
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
    ) -> npt.NDArray[np.float64]:
        results = []

        for rxn in reactions:
            priority = rxn.GetProp("_Priority")
            if priority == "uncommon":
                priority_factor = 0.2
            else:
                priority_factor = 1.0
            subset = rxn.GetProp("_Subset")

            reactive_atoms = np.array(
                extract_smiles_for_soms(rxn.GetReactantTemplate(0))
            ).reshape(-1, 1)
            descriptors = self.vectorizer.transform(reactive_atoms)
            predictions = self.models[subset].predict_proba(descriptors)

            results.append(predictions[:, -1].max() * priority_factor)

        return np.asarray(results)
