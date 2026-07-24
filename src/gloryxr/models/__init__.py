from abc import ABC, abstractmethod

import numpy as np
from rdkit.Chem.rdChemReactions import ChemicalReaction

__all__ = ["ModelProvider"]


class ModelProvider(ABC):
    """Base class for GLORYxR model providers"""

    @abstractmethod
    def predict_proba(
        self,
        reactions: list[ChemicalReaction],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Predict probabilities for a list of metabolism reactions, tagged with SOMs."""
        ...
