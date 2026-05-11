
import math
import torch
import random
import numpy as np
from torch import nn, optim

from datetime import datetime
from dataclasses import dataclass, asdict
from tokenizer import Tokenizer
from self_attention import MultiHeadSelfAttention


class PositionwiseFFN(nn.Module):
    """
    The position-wise FFN that follows after the self-attention computation.
    Vectors are projected to 4x the dimensionality and then projected down
    again after relu application.
    """

    def __init__(self, vector_dim, dropout_prob) :
        super().__init__()
        self.fc1 = nn.Linear(vector_dim, 4*vector_dim, bias=True)
        self.fc2 = nn.Linear(4*vector_dim, vector_dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class Block(nn.Module):
    """
    Transformer encoder block.

    This version differs from the original version in  [Vaswani et al. NeurIPS 2017],
    and applies the LayerNorm before the self-attention, and before the FFN, as this
    has proved to be beneficial (see [Nguyen and Salazar 2019]).
    """

    def __init__(self, vector_dim, n_heads, block_size, dropout_prob):
        super().__init__()
        att_dim = vector_dim // n_heads
        self.attn = MultiHeadSelfAttention(vector_dim, n_heads, block_size, is_causal=True)
        self.ffn = PositionwiseFFN(vector_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(vector_dim)
        self.ln2 = nn.LayerNorm(vector_dim)

    def forward(self, x):
        x1 = self.ln1(x)
        x2 = x + self.dropout(self.attn(x1))
        x3 = self.ln2(x2)
        x4 = x2 + self.dropout(self.ffn(x3))
        return x4


# ============= Hyper-parameters for training ============== #

@dataclass
class Config :
    vocab_size: int = 5000  # This number should agree with the tokenizer
    number_of_transformer_blocks: int = 4
    number_of_attention_heads: int = 4
    vector_dim: int = 256
    block_size: int = 512
    dropout_prob: float = 0.1
    batch_size: int = 8
    learning_rate: float = 0.0005
    weight_decay: float = 0.000001
    no_of_epochs: int = 1


class TransformerModel(nn.Module):

    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.embed =  nn.Embedding(config.vocab_size, config.vector_dim)
        self.positional = nn.Parameter(torch.randn(1, config.block_size, config.vector_dim))
        modules = [Block(config.vector_dim,\
                         config.number_of_attention_heads,\
                         config.block_size,\
                         config.dropout_prob) for _ in range(config.number_of_transformer_blocks)]
        self.transformers = nn.ModuleList(modules)
        self.final = nn.Linear(config.vector_dim, config.vocab_size)

    def forward(self, x):

        # YOUR CODE HERE
        #print("x",x.shape)
        emb = self.embed(x)
        #print("emb",emb.shape)
        #print("posisional", self.positional.shape)
        S = emb.shape[1]
        a = emb + self.positional[:, :S, :]
        #print("a",a.shape)

        for _, block in enumerate(self.transformers):
            a = block(a)

        logits = self.final(a)
        
        return logits

    def generate(self, ids, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            ids = ids[-self.config.block_size:]  # Make sure that we don't input more that `block_size` tokens
            logits = self.forward(ids.unsqueeze(0)).squeeze(0)
            logits = logits[-1, :]  # We only want the last time step
            logits = logits / temperature
            if top_k is not None:
                # Only consider the top k classes 
                top_idx = torch.topk(logits, top_k)[1]
                mask = torch.ones(self.config.vocab_size, dtype=torch.bool)
                mask[top_idx] = False
                logits[mask] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            # Sample from the distribution and append the new token to the input
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat((ids, next_id), dim=0)
        return ids
        
    @classmethod
    def load(cls, checkpoint_path, device='cpu'):
        """
        Loads a model from a checkpoint file.
        Automatically reconstructs the config and model architecture.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = Config(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
    
        print(f"Model loaded from {checkpoint_path} (Epoch {checkpoint['epoch']}, iteration {checkpoint['iteration']})")
        return model
