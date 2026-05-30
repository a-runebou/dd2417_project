


from collections import Counter
from pathlib import Path
import pickle
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.spell_correction import get_candidates



TEST_CSV_PATH = Path("evaluation/eval.csv")
WORD_COUNTS_PATH = Path("model/word_counts.pkl")
OUTPUT_DIR = Path("spell_eval_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOW_DISTANCE_2 = True



def load_word_counts(path: Path) -> Counter:
    with open(path, "rb") as f:
        obj = pickle.load(f)
        
    if isinstance(obj, Counter):
        return obj
    
    if isinstance(obj, dict):
        return Counter(obj)
    raise ValueError(f"Expected Counter or dict, got {type(obj)}")



def evaluate_row(row, word_counts: Counter, max_candidates: int) -> dict:
    typed = str(row["typed"]).strip().lower()
    correct = str(row["correct"]).strip().lower()
    
    candidates = get_candidates(
        typed,
        word_counts=word_counts,
        max_candidates=max_candidates,
        allow_distance_2=ALLOW_DISTANCE_2
    )
    
    is_error = typed != correct
    has_suggestion = len(candidates) > 0
    correct_in_candidates = correct in candidates
    
    result = {
        "correct": correct,
        "is_error": is_error, 
        "has_suggestion": has_suggestion,
        "correct_in_candidates": correct_in_candidates, 
    }
    
    
    return result


def compute_confusion_stats(results_df: pd.DataFrame) -> dict:
    error_rows = results_df["is_error"]
    clean_rows = ~results_df["is_error"]
    
    tp = int((error_rows & results_df["correct_in_candidates"]).sum())
    fn = int((error_rows & ~results_df["correct_in_candidates"]).sum())
    fp = int((clean_rows & results_df["has_suggestion"]).sum()) 
    tn = int((clean_rows & ~results_df["has_suggestion"]).sum())
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0 
    accuracy = (tp + tn) / len(results_df) if len(results_df) > 0 else 0.0 
    
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn, 
        "precision": precision, "recall": recall, 
        "accuracy": accuracy
    }



def plot_confusion_matrix(stats: dict, output_path: Path):
    matrix = np.array([[stats["tp"], stats["fn"]], [stats["fp"], stats["tn"]]])
    descriptions = np.array([["True Positive", "False Negative"], ["False Positive", "True Negative"]])
    
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    
    ax.set_xticklabels(
        ["Predicted Positive\nCorrection found", "Predicted Negative\nCorrection not found"],
        fontsize=11
    )
    ax.set_yticklabels(
        ["Actual Positive\nMisspelled word", "Actual Negative\nAlready correct word"],
        fontsize=11
    )
    
    # Cell annotations 
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{descriptions[i, j]}\n\n{matrix[i, j]}",
                ha="center", va="center", fontsize=14, fontweight="bold")
        
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of examples", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight") 
    plt.close()



def main():
    word_counts = load_word_counts(WORD_COUNTS_PATH)
    eval_df = pd.read_csv(TEST_CSV_PATH)
    results = [evaluate_row(row, word_counts, 3) for _, row in eval_df.iterrows()]
    results_df = pd.DataFrame(results)
    stats = compute_confusion_stats(results_df)
    
    # Make plot 
    plot_confusion_matrix(stats, OUTPUT_DIR / "confusion_matrix.png")
    
    print("\nConfusion stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value}") 
        else:
            print(f"{key}: {value}")
    




if __name__ == "__main__":
    main()
    