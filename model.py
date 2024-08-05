import torch
import torch.nn as nn
import math


class ImputEmbeddings(nn.Module):
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
