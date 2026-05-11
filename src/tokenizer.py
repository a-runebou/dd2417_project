import re
import json
from collections import Counter, defaultdict


class Tokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = []
        self.ids = {} 

    def _pretokenize(self, text:str) -> list[str]:
        regexp = r" ?\w+'?\w*| ?[^\w ]"  # REPLACE WITH YOUR REGULAR EXPRESSION
        tokens = re.findall(regexp, text)

        return tokens


    def _generate_word_frequencies(self, words:list[str]) -> dict[tuple[str], int]:
        # YOUR CODE HERE
        return Counter(tuple(word) for word in words)  # REPLACE THIS EXPRESSION WITH YOUR CODE


    def _count_token_bigrams(self, word_freqs: dict) -> defaultdict(int):
        bigram_counts = defaultdict(int)

        # YOUR CODE HERE
        for token, freq in word_freqs.items():
            for i in range(len(token)-1):
                key = (token[i], token[i+1])
                bigram_counts[key] += freq

        return bigram_counts


    def _find_most_frequent_token_bigram(self, word_freqs:dict) -> tuple[str, str]:
        bigram_counts = self._count_token_bigrams(word_freqs)
        # If there are several most frequent bigrams, return the first one

        best_bigram = max(bigram_counts, key=bigram_counts.get)  # REPLACE WITH YOUR CODE

        return best_bigram


    def _merge_bigram(self, word_freqs: dict, best_bigram: tuple, new_token: str) -> dict[str, int]:
        new_word_freqs = {}

        for word_tuple, freq in word_freqs.items():

            # REPLACE WITH YOUR CODE
            tup = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == best_bigram:
                    tup.append(new_token)
                    i += 2
                else:
                    tup.append(word_tuple[i])
                    i += 1

            new_word_freqs[tuple(tup)] = freq

        return new_word_freqs


    def train(self, corpus_path:str):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        words = self._pretokenize(raw_text)
        word_freqs = self._generate_word_frequencies(words)

        # Initialize vocabulary with all unique individual characters found
        unique_chars = set()
        for word_tuple in word_freqs:
            for char in word_tuple:
                unique_chars.add(char)
        self.vocab = list(unique_chars)

        # Merging loop
        merges = {}  # (bigram) -> priority (lower number -> higher priority)
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            best_bigram = self._find_most_frequent_token_bigram(word_freqs)   
            self.merges[best_bigram] = i # Rank         
            new_token = "".join(best_bigram)
            self.vocab.append(new_token)
            word_freqs = self._merge_bigram(word_freqs, best_bigram, new_token)
            if (i + 1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_bigram} -> {new_token}")
        print(f"Merge {i+1}/{num_merges}: {best_bigram} -> {new_token}")
        self.ids = {v: k for k, v in enumerate(self.vocab)}


    def tokenize(self, text):
        tokens = []
        pretokens = self._pretokenize(text)
        for word in pretokens:
            parts = list(word)
            while len(parts) > 1:

                # REPLACE WITH YOUR CODE
                possible_bigrams = [(parts[i], parts[i+1]) for i in range(len(parts)-1)]
                valid_bigrams = [bigram for bigram in possible_bigrams if bigram in self.merges.keys()]

                if not valid_bigrams:
                    break

                chosen_merge = min(valid_bigrams, key=lambda x: self.merges[x])
                new_token = "".join(chosen_merge)

                new_parts = []
                i = 0                
                while i < len(parts):
                    if i < len(parts) - 1 and (parts[i], parts[i+1]) == chosen_merge:
                        new_parts.append(new_token)
                        i += 2
                    else:
                        new_parts.append(parts[i])
                        i += 1

                parts = new_parts


            tokens.extend(parts)
        return tokens, [self.ids[t] for t in tokens]


    def save(self, path):
        serializable_merges = {f"{k[0]}<SPLIT>{k[1]}": v for k, v in self.merges.items()}
        data = {"vocab": self.vocab, "merges": serializable_merges, "vocab_size": self.vocab_size, "ids": self.ids}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)


    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls(vocab_size=data["vocab_size"])
        instance.vocab = data["vocab"]
        instance.ids = data["ids"]
        instance.merges = {tuple(k.split("<SPLIT>")): v for k, v in data["merges"].items()}
        return instance
