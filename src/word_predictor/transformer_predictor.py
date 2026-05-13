import math
import torch

from utils.text_state import split_context_prefix


class TransformerWordPredictor:
    """
    Transformer language model -> whole-word predictor
    
    The prefix is used as a hard filter.
    The Transformer scores complete candidate words in context.
    """
    def __init__(
        self,
        model,
        tokenizer,
        word_counts,
        max_candidates: int = 500,
        frequency_bonus: float = 0.02,
        lowercase_candidates: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.word_counts = word_counts
        self.max_candidates = max_candidates
        self.frequency_bonus = frequency_bonus
        self.lowercase_candidates = lowercase_candidates
        self.device = next(model.parameters()).device
        self.model.eval()
    
    def predict(self, text: str, k: int = 5) -> list[str]:
        #TODO: Begränsa context window size?
        context, prefix = split_context_prefix(text)
        candidates = self.get_candidates(prefix)
        
        if not candidates:
            return []
        
        scored = [] 
        for candidate in candidates:
            try:
                score = self.score_candidate(context, candidate)
            except KeyError:
                continue
            
            scored.append((candidate, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [word for word, score in scored[:k]]
    
    def get_candidates(self, prefix: str) -> list[str]:
        if self.lowercase_candidates:
            prefix_cmp = prefix.lower()
        else:
            prefix_cmp = prefix
        
        if prefix_cmp == "": 
            candidates = list(self.word_counts.keys())
        else:
            candidates = [word for word in self.word_counts.keys() if word.startswith(prefix_cmp)]
        
        candidates.sort(key=lambda word: self.word_counts[word], reverse=True) 
        return candidates[:self.max_candidates]

    def score_candidate(self, context: str, candidate: str) -> float:
        """
        Scores candidate as a complete word after context 
        """
        full_text = context + candidate + " "
        _, context_ids = self.tokenizer.tokenize(context)
        _, full_ids = self.tokenizer.tokenize(full_text)
        
        if len(full_ids) < 2:
            return float("-inf")
        
        full_tensor = torch.tensor(full_ids, dtype=torch.long, device=self.device)
        input_ids = full_tensor[:-1].unsqueeze(0)
        with torch.no_grad():
            logits = self.model(input_ids).squeeze(0)
            
        log_probs = torch.log_softmax(logits, dim=-1)
        if len(context_ids) == 0:
            start_pos = 0
        else:
            start_pos = len(context_ids) - 1
            
        score = 0.0
        scored_tokens = 0
        
        for pos in range(start_pos, len(full_ids) - 1):
            target_id = full_tensor[pos + 1]
            score += log_probs[pos, target_id].item()
            scored_tokens += 1
        
        if scored_tokens == 0:
            return float("-inf")
        
        score = score / scored_tokens
        count = self.word_counts.get(candidate, 1)
        score += self.frequency_bonus * math.log(count + 1)
        
        return score