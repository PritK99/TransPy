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
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,target_mask))
        x = self.residual_connections[1](x, self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embeddings: InputEmbeddings, target_embeddings: InputEmbeddings, src_pos: PositionalEncoding, target_pos: PositionalEncoding, projection: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embeddings
        self.target_embeddings = target_embeddings
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection = projection

    def encode (self, src, src_mask):
        src = self.src_embeddings(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode (self, encoder_output, target, src_mask, target_mask):
        target = self.target_embeddings(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)
    
    def project (self, x):
        return self.projection(x)

def build_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int, embedding_dim: int = 512, num_blocks: int = 6, num_heads: int = 8, dropout: float = 0.1, d_ff: int = 2048):

    src_embedding = InputEmbeddings(embedding_dim, src_vocab_size)
    target_embedding = InputEmbeddings(embedding_dim, target_vocab_size)
    src_pos = PositionalEncoding(embedding_dim, src_seq_len, dropout)
    target_pos = PositionalEncoding(embedding_dim, target_seq_len, dropout)

    encoder_blocks = []

    for _ in range (num_blocks):
        self_attention_block = MultiHeadAttentionBlock(embedding_dim, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(embedding_dim, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    decoder_blocks = []

    for _ in range (num_blocks):
        self_attention_block = MultiHeadAttentionBlock(embedding_dim, num_heads, dropout)
        cross_attention_block = MultiHeadAttentionBlock(embedding_dim, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(embedding_dim, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(embedding_dim, target_vocab_size)

    transformer = Transformer(encoder, decoder, src_embedding, target_embedding, src_pos, target_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer