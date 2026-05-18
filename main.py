from gui.app import App
from predictor.ngram_predictor import NgramPredictor
from predictor.transformer_predictor import TransformerPredictor


ngram = NgramPredictor("data/data_file.txt")
transformer = TransformerPredictor()
predictors = {
    'ngram': ngram,
    'transformer': transformer,
}


app = App(predictors)
app.run()