import argparse
import os

from tokenizer import Tokenizer
from text_utils import read_and_clean_text_files

def parse_args():
    parser = argparse.ArgumentParser(description="Train tokenizer from raw text files.")
    parser.add_argument("--texts", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="data/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--tmp-corpus", type=str, default="data/corpus.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    text = read_and_clean_text_files(args.texts)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.tmp_corpus), exist_ok=True)
    with open(args.tmp_corpus, "w", encoding="utf-8") as f:
        f.write(text)
        
    tokenizer = Tokenizer(vocab_size=args.vocab_size)
    tokenizer.train(args.tmp_corpus)
    tokenizer.save(args.output)