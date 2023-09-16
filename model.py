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
    
class FeedForwardBlock(nn.Module):
    def __init__(self, embedding_dim, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, embedding_dim)
    
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        assert ((embedding_dim % num_heads) == 0), "Embedding size is not divisible by number of heads"

        self.d_k = embedding_dim // num_heads

        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)

        self.w_o = nn.Linear(embedding_dim, embedding_dim)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query[-1]

        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if (mask is not None):
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if (dropout is not None):
            attention_scores = dropout(attention_scores)

        return (attention_scores@value) , attention_scores

    def forward(self, k, q, v, mask = None):

        # (batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim)
        key = self.w_k(k)
        query = self.w_q(q)
        value = self.w_v(v)
        
        # (batch, seq_len, embedding_dim) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2)

        x, self.attention_scoress = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contigous().view(x.shape[0],-1,self.num_heads*self.d_k)

        return self.w_o(x)
    
class ResidualConnection(nn.module):
    def __init__(self, dropout):
        super().__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))