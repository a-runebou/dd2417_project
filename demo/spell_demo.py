
from utils.word_vocabulary import load_word_counts
from utils.spell_correction import get_candidates


WORD_COUNTS_PATH = "model/word_counts.pkl"


def main():
    word_counts = load_word_counts(WORD_COUNTS_PATH)
    
    while True:
        prefix = input("\nTyped word/prefix: ").strip()
        
        if prefix == "":
            continue
        
        suggestions = get_candidates(
            word=prefix,
            word_counts=word_counts,
            max_candidates=10,
            allow_distance_2=True
        )
        print("\nCorrection candidates:")
        if not suggestions:
            print(" No candidates found.")
        else:
            for suggestion in suggestions:
                count = word_counts.get(suggestion, 0)
                print(f" {suggestion} count={count}")
        

if __name__ == "__main__":
    main()