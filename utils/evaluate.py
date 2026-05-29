from predictor.ngram_predictor import NgramPredictor
from predictor.transformer_score_predictor import TransformerScorePredictor
from transformer.tokenizer import Tokenizer
from transformer.transformer_lm import TransformerModel
from utils.word_vocabulary import load_word_counts

import nltk
import argparse
import sys
from tqdm import tqdm
import torch


def evaluate(predictor, sentence, num_preds: int = 3):
    try:
        words = nltk.word_tokenize(sentence)
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        words = nltk.word_tokenize(sentence)

    keystrokes_without = sum(len(w) for w in words) + (len(words) - 1)

    keystrokes_with = 0
    typed_this_far = ""

    for w_i, word in enumerate(words):
        found_prediction = False

        for c_i in range(len(word)):
            predictions = predictor.predict(typed_this_far, num_preds)

            if word in predictions:
                typed_this_far += word[c_i:] + " "
                keystrokes_with += 1

                found_prediction = True
                break
            else:
                typed_this_far += word[c_i]
                keystrokes_with += 1
        
        if not found_prediction and w_i != len(words)-1:
            typed_this_far += " "
            keystrokes_with += 1
    
    return keystrokes_with, keystrokes_without


def run_evaluate(filepath: str, predictor, num_preds: int = 3, output_path: str = None, predictor_name: str = "Unknown"):
    total_with = 0
    total_without = 0
    sentence_results = []

    out = open(output_path, 'w', encoding='utf-8') if output_path else sys.stdout

    try:
        print("=" * 50, file=out)
        print("Evaluation Setup", file=out)
        print(f"  Predictor  : {predictor_name}", file=out)
        print(f"  Test set   : {filepath}", file=out)
        print(f"  Num preds  : {num_preds}", file=out)
        print("=" * 50, file=out)
        print(file=out)

        with open(filepath, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="hej"):
                line = line.strip()
                if not line:
                    continue

                try:
                    sentences = nltk.sent_tokenize(line)
                except LookupError:
                    nltk.download('punkt')
                    sentences = nltk.sent_tokenize(line)

                for sentence in sentences:
                    ks_with, ks_without = evaluate(predictor, sentence, num_preds)
                    saved = ks_without - ks_with
                    percentage = (saved / ks_without * 100) if ks_without > 0 else 0.0

                    sentence_results.append((sentence, ks_with, ks_without, saved, percentage))
                    total_with += ks_with
                    total_without += ks_without

                    print(f"Sentence : {sentence}", file=out)
                    print(f"  Keystrokes without predictor : {ks_without}", file=out)
                    print(f"  Keystrokes with predictor    : {ks_with}", file=out)
                    print(f"  Keystrokes saved             : {saved}", file=out)
                    print(f"  Reduction                    : {percentage:.1f}%", file=out)
                    print(file=out)

        total_saved = total_without - total_with
        overall_percentage = (total_saved / total_without * 100) if total_without > 0 else 0.0
        avg_saved = (sum(r[3] for r in sentence_results) / len(sentence_results)) if sentence_results else 0.0
        avg_percentage = (sum(r[4] for r in sentence_results) / len(sentence_results)) if sentence_results else 0.0

        print("=" * 50, file=out)
        print(f"Sentences evaluated       : {len(sentence_results)}", file=out)
        print(f"Total keystrokes (no pred): {total_without}", file=out)
        print(f"Total keystrokes (pred)   : {total_with}", file=out)
        print(f"Total keystrokes saved    : {total_saved}", file=out)
        print(f"Overall reduction         : {overall_percentage:.1f}%", file=out)
        print(f"Avg keystrokes saved/sent : {avg_saved:.1f}", file=out)
        print(f"Avg reduction/sentence    : {avg_percentage:.1f}%", file=out)
        print("=" * 50, file=out)
    finally:
        if output_path:
            out.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", help="File name of test file.")
    parser.add_argument("-p", "--predictor", help="Predictor to use during testing.")
    parser.add_argument("-o", "--output", help="Path of the output file.")
    parser.add_argument("-k", "--num-preds", help="Number of predictions.", type=int)
    args = parser.parse_args()

    f_path = args.file_name
    if not f_path:
        print("Error: No file path provided.")
        return
    o_path = args.output
    num_preds = args.num_preds if args.num_preds else 3

    if args.predictor == "ngram" or args.predictor == 0:
        predictor = NgramPredictor("data/frankenstein")
        predictor_name = "Ngram"
    elif args.predictor == "transformer" or args.predictor == 1:
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
        predictor = transformer
        predictor_name = "Transformer"
    else:
        print("Error: No predictor provided.")
        return

    run_evaluate(f_path, predictor, num_preds=num_preds, output_path=o_path, predictor_name=predictor_name)


if __name__ == '__main__':
    main()
