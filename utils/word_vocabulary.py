import re
import pickle
from collections import Counter
from utils.text_utils import clean_text


def extract_words_from_text(text: str, lowercase: bool = True) -> Counter:
    """
    Extracts whole-word candidates from text 
    """
    text = clean_text(text)
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if lowercase:
        words = [word.lower() for word in words]
    return Counter(words)


def build_word_counts_from_file(
    corpus_path: str,
    min_count: int = 0,
    lowercase: bool = True
) -> Counter:
    with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    
    counts = extract_words_from_text(text, lowercase=lowercase)
    counts = Counter({
        word: count
        for word, count in counts.items()
        if count >= min_count
    })
    return counts


def build_word_counts_from_files(
    corpus_paths: list[str],
    min_count: int = 2,
    lowercase: bool = True
) -> Counter:
    total = Counter()
    
    for path in corpus_paths:
        total.update(build_word_counts_from_file(path, min_count=1, lowercase=lowercase))
    
    total = Counter({
        word: count
        for word, count in total.items()
        if count >= min_count
    })
    return total


def save_word_counts(word_counts: Counter, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(word_counts, f)


def load_word_counts(path: str) -> Counter:
    with open(path, "rb") as f:
        return pickle.load(f)
    
