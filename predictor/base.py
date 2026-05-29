from abc import ABC, abstractmethod

class BasePredictor(ABC):
    @abstractmethod
    def predict(self, text: str, k: int = 3) -> list[str]:
        """Given the current text, give k predictions of the next word"""
        pass

    @abstractmethod
    def load(self):
        """Load the predictor into memory"""
        pass
