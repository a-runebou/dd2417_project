from gui.app import App
from predictor.ngram_predictor import NgramPredictor
from predictor.transformer_score_predictor import TransformerScorePredictor

import torch
from transformer.tokenizer import Tokenizer
from transformer.transformer_lm import TransformerModel
from utils.word_vocabulary import load_word_counts

ngram = NgramPredictor("data/frankenstein")


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "model/checkpoint.pt"
tokenizer_path = "model/tokenizer.json"
word_counts_path = "model/word_counts.pkl"

model = TransformerModel.load(checkpoint_path, device=device)
tokenizer = Tokenizer.load(tokenizer_path)
word_counts = load_word_counts(word_counts_path)

transformer = TransformerScorePredictor(
    model=model,
    tokenizer=tokenizer,
    word_counts=word_counts,
    max_candidates=300,
    frequency_bonus=0.0
)
predictors = {
    'ngram': ngram,
    'transformer': transformer,
}


app = App(predictors)
app.run()