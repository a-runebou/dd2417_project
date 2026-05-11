from abc import ABC, abstractmethod

class BasePredictor(ABC):
    @abstractmethod
    def predict(self, context: str, n: int = 3):
        """Given the current context, give n predictions of the next word"""
        pass

    @abstractmethod
    def load(self):
        """Load the predictor into memory"""
        pass
