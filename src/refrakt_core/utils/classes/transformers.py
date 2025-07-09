"""
Transformer components for embedding input tokens and applying positional encoding.

Modules:
- InputEmbeddings: Embeds token indices into dense vectors.
- PositionalEncoding: Adds sinusoidal positional information to embeddings.
"""

import math
from typing import cast

import torch
from torch import nn


class InputEmbeddings(nn.Module):
    """
    Converts token indices into dense vector embeddings and scales them.

    Args:
        d_model (int): Embedding dimension.
        vocab_size (int): Size of the vocabulary.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.vocab_size: int = vocab_size
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes scaled token embeddings.

        Args:
            x (Tensor): Input Tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor: Scaled embeddings of shape (batch_size, sequence_length, d_model).
        """
        return cast(torch.Tensor, self.embedding(x) * math.sqrt(self.d_model))


class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encodings to token embeddings.

    Args:
        d_model (int): Embedding dimension.
        seq_len (int): Maximum sequence length.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.seq_len: int = seq_len
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        # Precompute positional encodings
        pe: torch.Tensor = torch.zeros(seq_len, d_model)
        position: torch.Tensor = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )
        denom: torch.Tensor = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * denom)
        pe[:, 1::2] = torch.cos(position * denom)

        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the input torch.Tensor.

        Args:
            x (Tensor): Input Tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            Tensor: Positionally encoded Tensor of same shape.
        """
        x = (
            x + self.pe[:, : x.shape[1], :].detach()
        )  # Ensure buffer stays non-trainable
        return cast(torch.Tensor, self.dropout(x))
