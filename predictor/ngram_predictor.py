from predictor.base import BasePredictor

class NgramPredictor(BasePredictor):
    def __init__(self):
        pass

    def predict(self, context: str, n: int = 3):
        """Given the current context, give n predictions of the next word"""
        pass

    def load(self):
        """Load the predictor into memory"""
        pass
