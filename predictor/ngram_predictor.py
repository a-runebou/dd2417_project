from predictor.base import BasePredictor

class NgramPredictor(BasePredictor):
    def __init__(self):
        pass

    def predict(self, text: str, k: int = 3)  -> list[str]:
        """Given the current text, give k predictions of the next word"""
        return []

    def load(self):
        """Load the predictor into memory"""
        pass
