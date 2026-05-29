from predictor.base import BasePredictor
from collections import defaultdict, Counter
import nltk
import codecs

class NgramPredictor(BasePredictor):
    def __init__(self, corpus_path, max_n=3):
        self.max_n = max_n

        self.index = {}
        self.word = {}

        self.models = {
            n: defaultdict(Counter) for n in range(1, max_n+1)
        }

        self.history = []

        self._process_files(corpus_path)

    def predict(self, text: str, k: int = 3)  -> list[str]:
        """Given the current text, give k predictions of the next word"""
        tokens = nltk.word_tokenize(text.lower())

        if not tokens:
            return []

        if text[-1] == " ":
            partial = ""
            context_tokens = tokens
        else:
            partial = tokens[-1]
            context_tokens = tokens[:-1]

        for level in range(self.max_n, 0, -1):
            context_key = tuple(context_tokens[-(level - 1):]) if level > 1 else ()
            if context_key in self.models[level]:
                candidates = self.models[level][context_key].most_common()
                filtered = [w for w, _ in candidates if w.startswith(partial)]
                if filtered:
                    return filtered[:k]

        return []

    def load(self):
        """Load the predictor into memory"""
        pass


    def _process_files(self, fname):
        with codecs.open(fname, 'r', 'utf-8') as text_file:
            text = reader = text_file.read().encode('utf-8').decode().lower()
        try :
            tokens = nltk.word_tokenize(text) 
        except LookupError :
            nltk.download('punkt')
            nltk.download('punkt_tab')
            tokens = nltk.word_tokenize(text)
    
        self._build_model(tokens)
        
    def _build_model(self, tokens):
        for n in range(1, self.max_n + 1):
            for i in range(len(tokens) - n + 1):
                context = tuple(tokens[i : i + n - 1])
                next_word = tokens[i + n - 1]
                self.models[n][context][next_word] += 1    
