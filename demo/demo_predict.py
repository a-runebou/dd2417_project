import torch

from transformer.tokenizer import Tokenizer
from transformer.transformer_lm import TransformerModel
from utils.word_vocabulary import load_word_counts
from predictor.transformer_score_predictor import TransformerScorePredictor
from predictor.transformer_generate_predictor import TransformerGeneratePredictor


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "model/checkpoint.pt"
tokenizer_path = "model/tokenizer.json"
word_counts_path = "model/word_counts.pkl"

model = TransformerModel.load(checkpoint_path, device=device)
tokenizer = Tokenizer.load(tokenizer_path)
word_counts = load_word_counts(word_counts_path)

predictor1 = TransformerScorePredictor(
    model=model,
    tokenizer=tokenizer,
    word_counts=word_counts,
    max_candidates=300,
    frequency_bonus=0.0
)

predictor2 = TransformerGeneratePredictor(
    model=model,
    tokenizer=tokenizer,
    max_attempts=500,
    max_new_tokens=16,
    temperature=0.9,
    top_k=40
)


while True:
    text = input("\nTyped text: ")
    
    if text.lower() in {"quit", "exit"}:
        break
    
    suggestions1 = predictor1.predict(text, k=5)
    suggestions2 = predictor2.predict(text, k=5)
    print("Suggestions from predictor1:")
    for s in suggestions1:
        print("  ", s)
    print("Suggestions from predictor2:")
    for s in suggestions2:
        print("  ", s)