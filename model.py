import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    Class to handle the InputEmbeddings, Covert the input sequence to vector embeddings
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        '''Multiply embedding by sqrt(d_model) as per the paper'''
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    '''
    Class to give information rearding the position of the embedding wrt to the input sequence
    '''

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # A matrix of shape(seq_len,d_model)
        pe = torch.zeros(seq_len, d_model)
        # Represents position of the word in sentence, vector of shape(seq_len,1)
        position = torch.arrange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arrange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # apply sin to even position and cos to odd.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # to get batch size in dimension
        pe = pe.unsqueeze(0)

        # To save the tensor to the file along with the state of the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requries_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # To multiply
        self.bias = nn.Parameter(torch.zeros(1))  # to add

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std - self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and b2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, 'd_model not divisible by h'

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (batch,h,seq_len,d_k) -> (batch,h,seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch,seq_len,d_K) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.h, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        value = value.view(
            value.shape[0], value.shape[1], self.h, self.d_k
        ).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout
        )

        # (batch,h,seq_len,d_k) -> (batch,seq_len,h,d_k) -> (batch,seq_len,d_model)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.h * self.d_k)
        )

        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Droput(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(
        self,
        self_attention: MultiHeadAttention,
        feed_forward: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, c, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feed_forward: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, trgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, trgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.self_attention(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, trgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        trgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        trgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trgt_embed = trgt_embed
        self.src_pos = src_pos
        self.trgt_pos = trgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, trgt, trgt_mask):
        trgt = self.trgt_embed(trgt)
        trgt = self.trgt_pos(trgt)
        return self.decoder(trgt, encoder_output, src_mask, trgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    trgt_vocab_size: int,
    src_seq_len: int,
    trgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    # Embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    trgt_embed = InputEmbeddings(d_model, trgt_vocab_size)

    # Positional Encoding Layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trgt_pos = PositionalEncoding(d_model, trgt_seq_len, dropout)

    # Encoder block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention, feed_forward, dropout
        )
        encoder_blocks.append(encoder_block)

    # Decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention,
            decoder_cross_attention,
            feed_forward,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # create Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Projection layer
    projection_layer = ProjectionLayer(d_model, trgt_vocab_size)

    # The Transformer
    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        trgt_embed,
        src_pos,
        trgt_pos,
        projection_layer,
    )

    # Initialize parameters

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
