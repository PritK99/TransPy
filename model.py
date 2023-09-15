import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embeddings(x)*math.sqrt(self.embedding_dim)

