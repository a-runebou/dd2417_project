import torch

from tokenizer import Tokenizer
from transformer_lm import TransformerModel
from word_predictor.word_vocabulary import load_word_counts
from word_predictor.transformer_predictor import TransformerWordPredictor


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "data/checkpoint.pt"
tokenizer_path = "data/tokenizer.json"
word_counts_path = "data/word_counts.pkl"

model = TransformerModel.load(checkpoint_path, device=device)
tokenizer = Tokenizer.load(tokenizer_path)
word_counts = load_word_counts(word_counts_path)

predictor = TransformerWordPredictor(
    model=model,
    tokenizer=tokenizer,
    word_counts=word_counts,
    max_candidates=300,
    frequency_bonus=0.0
)


while True:
    text = input("\nTyped text: ")
    
    if text.lower() in {"quit", "exit"}:
        break
    
    suggestions = predictor.predict(text, k=5)
    print("Suggestions:")
    for s in suggestions:
        print("  ", s)