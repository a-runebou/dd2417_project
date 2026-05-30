
from collections import Counter


LETTERS = "abcdefghijklmnopqrstuvwxyz"

def _edits1(word: str) -> set[str]:
    """
    Returns all strings that are one edit away from word 
    Edits: Deletion, insertion, substitution, transposition
    """
    
    # Generate all possible splits of the word into a left and right part
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    
    deletions = {
        left + right[1:]
        for left, right in splits 
        if right
    }
    insertions = {
        left + c + right
        for left, right in splits 
        for c in LETTERS
    }
    substitutions = {
        left + c + right[1:]
        for left, right in splits 
        if right
        for c in LETTERS
    }
    transpositions = {
        left + right[1] + right[0] + right[2:]
        for left, right in splits 
        if len(right) > 1
    }
    
    return deletions | insertions | substitutions | transpositions 


def _known(words: set[str], vocabulary: set[str]) -> set[str]:
    """
    Filters generated words to words that exist in the vocabulary 
    """
    return words & vocabulary

def get_candidates(
    word: str,
    word_counts: Counter, 
    max_candidates: int = 100,
    allow_distance_2: bool = True) -> list[str]:
    """
    Generates spelling-correction candidates for a typed word 
    """
    word = word.lower()
    vocabulary = set(word_counts.keys())
    candidates = _known(_edits1(word), vocabulary)
    
    # Edit distance 2 if allowed
    if not candidates and allow_distance_2:
        candidates2 = set()
        for e1 in _edits1(word):
            candidates2.update(_known(_edits1(e1), vocabulary))
        candidates = candidates2
    
    # Sort candidates by freq 
    ranked = sorted(
        candidates,
        key=lambda word: word_counts[word],
        reverse=True
    )
    return ranked[:max_candidates]

