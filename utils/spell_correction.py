
from collections import Counter


LETTERS = "abcdefghijklmnopqrstuvwxyz"

def edits1(word: str) -> set[str]:
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


def known(words: set[str], vocabulary: set[str]) -> set[str]:
    """
    Filters generated words to words that exist in the vocabulary 
    """
    return words & vocabulary

def correction_candidates(
    prefix: str,
    word_counts: Counter, 
    max_candidates: int = 100,
    allow_distance_2: bool = True) -> list[str]:
    """
    Generates spelling-correction candidates for a typed prefix 
    """
    prefix = prefix.lower()
    vocabulary = set(word_counts.keys())
    candidates = known(edits1(prefix), vocabulary)
    
    # Edit distance 2 if allowed
    if not candidates and allow_distance_2:
        candidates2 = set()
        for e1 in edits1(prefix):
            candidates2.update(known(edits1(e1), vocabulary))
        candidates = candidates2
    
    # Sort candidates by freq 
    ranked = sorted(
        candidates,
        key=lambda word: word_counts[word],
        reverse=True
    )
    return ranked[:max_candidates]

