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

class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim, seq_len, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Creating the positional encoding matrix of shape (seq_len, embedding_dim)
        pe = torch.zeros(seq_len, embedding_dim)

        # Creating a vector of length (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0,seq_len,2).float()*(-math.log(10000.0) / embedding_dim))

        pe[:, ::2] = torch.sin(position*denominator)
        pe[:, 1::2] = torch.cos(position*denominator)

        # Converting (seq_len, embedding_dim) to (1, seq_len, embedding_dim)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
    
    def forward(self, x):

        # Adding embeddings with positional encodings. 
        # x.shape[1] helps to add positional encodings for the positions that are present in the actual input sequence
        x = x + (self.pe[:, :x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, epsilon=10**-6):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return (self.alpha)*((x - mean)/(std + self.epsilon)) + self.beta
    
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, embedding_dim)
    
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
