import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Input Embeddings module for mapping input words to vectors.

    This module initializes an embedding layer for representing input sequences. It scales the embeddings by the square root of the embedding dimension.

    Parameters:
        - embedding_dim (int): The dimensionality of the input embeddings.
        - vocab_size (int): The size of the vocabulary, i.e., the maximum number of input tokens.

    Forward Input:
        - x (torch.Tensor): Input tensor representing the input tokens.

    Forward Output:
        - embeddings (torch.Tensor): Input embeddings scaled by sqrt(embedding_dim).
    """
    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embeddings(x)*math.sqrt(self.embedding_dim)

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for adding positional information to input sequences.

    This module generates and adds positional encodings to input sequences. The positional encodings are constructed based on the sine and cosine functions. The positional encodings are added to embeddings to enhance the input to encoder. Here, we use a modified formula involving exponentiation and logarithm for numerical stability.

    Parameters:
        - embedding_dim (int): The dimensionality of the input embeddings.
        - seq_len (int): The maximum sequence length for which positional encodings will be generated.
        - dropout (float): The dropout probability applied to the positional encodings.

    Attributes:
        - pe (torch.Tensor): The positional encoding matrix of shape (1, seq_len, embedding_dim). This is Not Learnable.

    Forward Input:
        - x (torch.Tensor): Input tensor representing the sequence embeddings.

    Forward Output:
        - x (torch.Tensor): Sequence embeddings with added positional encodings.
    """
    def __init__(self, embedding_dim: int, seq_len: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        pe = torch.zeros(seq_len, embedding_dim) # Creating the positional encoding matrix of shape (seq_len, embedding_dim)
        
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0,seq_len,2).float()*(-math.log(10000.0) / embedding_dim))

        pe[:, ::2] = torch.sin(position*denominator)
        pe[:, 1::2] = torch.cos(position*denominator)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1],:]).requires_grad_(False)
        return x

class LayerNorm(nn.Module):
    """
    Layer Normalization module for stabilizing training in neural networks.

    This module performs Layer Normalization on input tensors. It ensures that the mean of all activations for a given layer is approximately 0, and the standard deviation is approximately 1, helping in stable and efficient training.

    Parameters:
        - epsilon (float): A small constant added to the denominator for numerical stability.
    
    Attributes:
        - alpha (nn.Parameter): Learnable scaling parameter initialized to 1.
        - beta (nn.Parameter): Learnable shifting parameter initialized to 0.

    Forward Input:
        - x (torch.Tensor): Input tensor to be layer-normalized.

    Forward Output:
        - output (torch.Tensor): Layer-normalized tensor after scaling and shifting.
    """
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
    """
    Feed-Forward Block module for sequence-to-sequence models.

    This module represents a neural network block with two linear layers and a ReLU activation function. The feed-forward block processes input sequences by applying a linear transformation, a non-linear activation (ReLU), dropout for regularization, and another linear transformation. Feed forward block is simple neural network of 3 layers [embedding_dim, d_ff, embedding_dim]

    Parameters:
        - embedding_dim (int): The dimensionality of the input and output embeddings.
        - d_ff (int): The dimensionality of the intermediate (hidden) layer.
        - dropout (float): The dropout probability applied to the intermediate layer.

    Attributes:
        - linear_1 (nn.Linear): First linear layer transforming input to the hidden dimension.
        - dropout (nn.Dropout): Dropout layer for regularization.
        - linear_2 (nn.Linear): Second linear layer transforming the hidden dimension back to the original embedding dimension.

    Forward Input:
        - x (torch.Tensor): Input tensor representing sequence embeddings.

    Forward Output:
        - output (torch.Tensor): Sequence embeddings after passing through the feed-forward block.
    """
    def __init__(self, embedding_dim: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, embedding_dim)
    
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class ResidualConnection(nn.Module):
    """
    Residual Connection module for skip connections in architecture.

    Residual connections reduce vanishing gradients problem by allowing the direct flow of information from one layer to another. It consists of layer normalization, dropout for regularization, and the addition of the original input to the output of a sublayer.

    Parameters:
        - dropout (float): The dropout probability applied to the residual connection.

    Attributes:
        - norm (LayerNorm): Layer normalization module for normalizing the input.
        - dropout (nn.Dropout): Dropout layer for regularization.

    Forward Input:
        - x (torch.Tensor): Input tensor to be passed through the residual connection.
        - sublayer (nn.Module): Sublayer module to apply on the normalized and dropout-adjusted input.

    Forward Output:
        - output (torch.Tensor): Output tensor after applying the residual connection.
    """
    def __init__(self, dropout: float):
        super().__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention Block module for transformer architecture.

    The block comprises linear transformations for keys, queries, and values, along with an output linear layer. The attention mechanism is implemented through a static method. During the forward pass, input key, query, and value tensors are transformed and reshaped to accommodate multiple attention heads. The attention mechanism is applied, and the resulting tensor is reshaped and passed through an output linear layer.

    Parameters:
        - embedding_dim (int): Dimensionality of input and output embeddings.
        - num_heads (int): Number of attention heads.
        - dropout (float): Dropout probability applied to attention scores.

    Attributes:
        - embedding_dim (int): Dimensionality of input and output embeddings.
        - num_heads (int): Number of attention heads.
        - dropout (nn.Dropout): Dropout layer for attention scores.
        - d_k (int): Dimensionality of each head.
        - w_k (nn.Linear): Linear transformation for keys.
        - w_q (nn.Linear): Linear transformation for queries.
        - w_v (nn.Linear): Linear transformation for values.
        - w_o (nn.Linear): Linear transformation for the output.

    Static Methods:
        - attention(query, key, value, mask, dropout): Perform scaled dot-product attention.

    Forward Input:
        - k (torch.Tensor): Key tensor.
        - q (torch.Tensor): Query tensor.
        - v (torch.Tensor): Value tensor.
        - mask (torch.Tensor, optional): Mask to apply to attention scores.

    Forward Output:
        - output (torch.Tensor): Output tensor after multi-head attention.
    """
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
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

        return (attention_scores @ value) , attention_scores

    def forward(self, k, q, v, mask = None):
        key = self.w_k(k)
        query = self.w_q(q)
        value = self.w_v(v)
        
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2)     #transpose allows us to expose the dimensions (seq_len, d_k) which makes it convenient to deal with attention.

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)     #we call the static attention function

        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.num_heads*self.d_k)

        return self.w_o(x)

class EncoderBlock(nn.Module):
    """
    Encoder Block for transformer architecture.

    This represents a single encoder block which consists of a self-multihead-attention mechanism and a feed-forward neural network, each followed by a residual connection and Add & Norm layers. The residual connections act as Add & Norm along with skip connections and include dropout for regularization.

    Parameters:
        - self_attention_block (MultiHeadAttentionBlock): Self-attention block.
        - feed_forward_block (FeedForwardBlock): Feed-forward block.
        - dropout (float): Dropout probability applied to residual connections.

    Attributes:
        - self_attention_block (MultiHeadAttentionBlock): Self-attention block.
        - feed_forward_block (FeedForwardBlock): Feed-forward block.
        - residual_connections (nn.ModuleList): List of residual connection modules.

    Forward Input:
        - x (torch.Tensor): Input tensor.
        - src_mask (torch.Tensor): Mask for source sequence.

    Forward Output:
        - output (torch.Tensor): Output tensor after passing through the encoder block.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # We pass lambda function here which indicates x is taken as input and used as parameter. lambda function allows us to customize this for different cases, such as cross-attention-block which is defined in decoder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    """
    Encoder module composed of multiple stacked encoder blocks.

    Parameters:
        - layers (nn.ModuleList): List of encoder blocks.
    
    Attributes:
        - layers (nn.ModuleList): List of encoder blocks.
        - norm (LayerNorm): Layer normalization module.

    Forward Input:
        - x (torch.Tensor): Input tensor.
        - mask (torch.Tensor): Mask for the input sequence.

    Forward Output:
        - output (torch.Tensor): Output tensor after passing through the encoder.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    Decoder Block for transformer architecture.

    The block incorporates a self-attention mechanism, a cross-attention mechanism, and a feed-forward neural network, each followed by a residual connection. The residual connections include dropout for regularization. The decoder block is designed to process input data in a sequence-to-sequence manner, considering both the target sequence and information from the encoder. The self-attention mechanism attends to the target sequence, while the cross-attention mechanism incorporates information from the encoder's output. The block structure includes skip connections and normalization layers.

    Parameters:
        - self_attention_block (MultiHeadAttentionBlock): Self-attention block.
        - cross_attention_block (MultiHeadAttentionBlock): Cross-attention block.
        - feed_forward_block (FeedForwardBlock): Feed-forward block.
        - dropout (float): Dropout probability applied to residual connections.

    Attributes:
        - self_attention_block (MultiHeadAttentionBlock): Self-attention block.
        - cross_attention_block (MultiHeadAttentionBlock): Cross-attention block.
        - feed_forward_block (FeedForwardBlock): Feed-forward block.
        - residual_connections (nn.ModuleList): List of residual connection modules.

    Forward Input:
        - x (torch.Tensor): Input tensor.
        - encoder_output (torch.Tensor): Output from the encoder.
        - src_mask (torch.Tensor): Mask for the source sequence.
        - target_mask (torch.Tensor): Mask for the target sequence.

    Forward Output:
        - output (torch.Tensor): Output tensor after passing through the decoder block.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    """
    Decoder module composed of multiple stacked decoder blocks.

    Parameters:
        - layers (nn.ModuleList): List of decoder blocks.

    Attributes:
        - layers (nn.ModuleList): List of decoder blocks.
        - norm (LayerNorm): Layer normalization module.

    Forward Input:
        - x (torch.Tensor): Input tensor.
        - encoder_output (torch.Tensor): Output from the encoder.
        - src_mask (torch.Tensor): Mask for the source sequence.
        - target_mask (torch.Tensor): Mask for the target sequence.

    Forward Output:
        - output (torch.Tensor): Output tensor after passing through the decoder.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    Projection Layer for transformer architectures.

    The linear layer helps to output target word from given embedding. This layer includes a linear transformation to project the input tensor to the target vocabulary space. The output is processed through a log-softmax function.

    Parameters:
        - embedding_dim (int): Dimensionality of input embeddings.
        - vocab_size (int): Size of the target vocabulary.

    Attributes:
        - proj (nn.Linear): Linear transformation for projection.

    Forward Input:
        - x (torch.Tensor): Input tensor.

    Forward Output:
        - output (torch.Tensor): Log-softmax output after projection.
    """
    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    """
    Transformer architecture consisting of an encoder, decoder, and projection layer.

    Parameters:
        - encoder (Encoder): Encoder module.
        - decoder (Decoder): Decoder module.
        - src_embeddings (InputEmbeddings): Embedding layer for source sequences.
        - target_embeddings (InputEmbeddings): Embedding layer for target sequences.
        - src_pos (PositionalEncoding): Positional encoding for source sequences.
        - target_pos (PositionalEncoding): Positional encoding for target sequences.
        - projection (ProjectionLayer): Projection layer for final output.

    Forward Input (for encoding):
        - src (torch.Tensor): Source sequence tensor.
        - src_mask (torch.Tensor): Mask for source sequence.

    Forward Output (for encoding):
        - output (torch.Tensor): Encoded representation of the source sequence.

    Forward Input (for decoding):
        - encoder_output (torch.Tensor): Output from the encoder.
        - target (torch.Tensor): Target sequence tensor.
        - src_mask (torch.Tensor): Mask for the source sequence.
        - target_mask (torch.Tensor): Mask for the target sequence.

    Forward Output (for decoding):
        - output (torch.Tensor): Decoded representation of the target sequence.

    Forward Input (for projection):
        - x (torch.Tensor): Input tensor.

    Forward Output (for projection):
        - output (torch.Tensor): Log-softmax output after projection.
    """
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
    """
    Builds a transformer model with specified configurations. This is the driver function which will be used by other files to build transformer object.

    Parameters:
        - src_vocab_size (int): Size of the source vocabulary.
        - target_vocab_size (int): Size of the target vocabulary.
        - src_seq_len (int): Maximum length of the source sequence.
        - target_seq_len (int): Maximum length of the target sequence.
        - embedding_dim (int): Dimensionality of the embedding layer.
        - num_blocks (int): Number of encoder and decoder blocks.
        - num_heads (int): Number of attention heads.
        - dropout (float): Dropout probability applied throughout the model.
        - d_ff (int): Dimensionality of the feed-forward layer.

    Returns:
        - transformer (Transformer): Transformer model.
    """
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

    # xavier_uniform allows us initialize our matrix parameters such that it reduces vanishing gradient problem
    for p in transformer.parameters():
        if p.dim > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer