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
