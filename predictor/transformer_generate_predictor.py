import math
import re
from typing import Counter
import torch

from utils.text_state import split_context_prefix

from predictor.base import BasePredictor

class TransformerGeneratePredictor(BasePredictor):
    """
    Transformer language model -> whole-word predictor
    
    The context + prefix is used as a prompt.
    The Transformer generates k unique candidate words from the prompt.
    """
    def __init__(
        self,
        model,
        tokenizer,
        max_attempts: int = 500,
        max_new_tokens: int = 16,
        temperature: float = 0.9,
        top_k: int = 40
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_attempts = max_attempts
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.device = next(model.parameters()).device
        self.model.eval()

    def predict(self, text: str, k: int = 3) -> list[str]:
        """Given the current text, give k predictions of the next word"""
        #TODO: Begränsa context window size?
        context, prefix = split_context_prefix(text)
        prompt = context + prefix
        
        # Tokenize the prompt
        try:
            _, prompt_ids = self.tokenizer.tokenize(prompt)
        except KeyError:
            return []
        
        if len(prompt_ids) == 0:
            return []
        
        
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        suggestions = Counter()
        attempts = 0
        
        # We keep generating until we have k unique suggestions 
        # or we have reached the maximum number of attempts
        while len(suggestions) < k and attempts < self.max_attempts:
            attempts += 1
            with torch.no_grad():
                generated_ids = self.model.generate(
                    prompt_tensor.clone(),
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k
                )
            
            generated_text = self.decode_ids(generated_ids.tolist())
            decoded_prompt = self.decode_ids(prompt_ids)
            continuation = generated_text[len(decoded_prompt):]
            
            word = self.extract_generated_word(prefix=prefix, continuation=continuation)
            if word is None:
                continue
            
            suggestions[word] += 1
        
        return [word for word, count in suggestions.most_common(k)]
    
    def extract_generated_word(self, prefix: str, continuation: str):
        """
        Extracts the first whole word from prefix + continuation.
        """
        
        raw = prefix + continuation
        if prefix == "":
            raw = raw.lstrip()
        
        match = re.match(r"[A-Za-z]+(?:'[A-Za-z]+)?", raw)
        if match is None:
            return None
        
        word = match.group(0).lower()
        
        # Filter out bad predictions
        if prefix != "" and not word.startswith(prefix.lower()):
            return None
        if prefix != "" and word == prefix.lower():
            return None
        
        return word
    
    def decode_ids(self, ids: list[int]) -> str:
        return "".join(self.tokenizer.vocab[token_id] for token_id in ids)
    
    def load(self):
        """Load the predictor into memory"""
        pass