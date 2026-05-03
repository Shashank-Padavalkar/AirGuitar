from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class BaseDetector(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> Any:
        pass
        
    @abstractmethod
    def close(self):
        pass
