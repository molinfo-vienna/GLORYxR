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
    ) -> npt.NDArray[np.float_]: ...


class _LocalModelProvider(_ModelProvider):
    def __init__(self) -> None:
        models_path = Path("models")

        self.models: dict[str, Any] = {}
        for model_path in models_path.glob("*.joblib"):
            self.models[model_path.stem] = joblib.load(filename=model_path)

    @override
    def predict_proba(
        self,
        subset: str,
        descriptors: npt.ArrayLike,
    ) -> npt.NDArray[np.float_]:
        return self.models[subset].predict_proba(descriptors)
