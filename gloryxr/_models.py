from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, override

import joblib
import numpy as np
import numpy.typing as npt


class _ModelProvider(ABC):
    @abstractmethod
    def predict_proba(
        self,
        subset: str,
        descriptors: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]: ...


class _LocalModelProvider(_ModelProvider):
    def __init__(self, phase: int) -> None:
        """
        Initialize the local model provider.

        Args:
            phase: Metabolism phase (1, 2, or 3 for both)
        """
        models_path = Path("models")

        self.models: dict[str, Any] = {}
        for model_path in models_path.glob("*.joblib"):
            if phase == 3:
                self.models[model_path.stem] = joblib.load(filename=model_path)
            elif phase == 1:
                if "phase 1" in model_path.stem.lower():
                    self.models[model_path.stem] = joblib.load(
                        filename=model_path
                    )
            elif phase == 2:
                if "phase 2" in model_path.stem.lower():
                    self.models[model_path.stem] = joblib.load(
                        filename=model_path
                    )

    @override
    def predict_proba(
        self,
        subset: str,
        descriptors: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return self.models[subset].predict_proba(descriptors)
