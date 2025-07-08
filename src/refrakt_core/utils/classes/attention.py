"""Attention modules including Multi-Head Attention (MHA), Multi-Head Self-Attention (MSA),
and Shifted Window MSA, used across transformer and vision transformer models."""

import math
from typing import Tuple, cast

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional

from refrakt_core.utils.classes.embedding import RelativeEmbedding


class MSA(nn.Module):
    """
    Multi-Head Self-Attention (non-batched variant).

    Args:
        d (int): Input embedding dimension.
        n_heads (int): Number of attention heads.
    """

    def __init__(self, d: int, n_heads: int = 4) -> None:
        super().__init__()
        assert d % n_heads == 0, "Embedding dimension must be divisible by n_heads"
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.q_mappings = nn.ModuleList(
            [nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)]
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences: Tensor) -> Tensor:
        """
        Applies self-attention across all sequences.

        Args:
            sequences (Tensor): Tensor of shape (batch, seq_len, d)

        Returns:
            Tensor: Output tensor after self-attention.
        """
        result = []
        for sequence in sequences:
            head_outputs = []
            for head in range(self.n_heads):
                q = self.q_mappings[head](
                    sequence[:, head * self.d_head : (head + 1) * self.d_head]
                )
                k = self.k_mappings[head](
                    sequence[:, head * self.d_head : (head + 1) * self.d_head]
                )
                v = self.v_mappings[head](
                    sequence[:, head * self.d_head : (head + 1) * self.d_head]
                )
                attn = self.softmax(q @ k.T / math.sqrt(self.d_head))
                head_outputs.append(attn @ v)
            result.append(torch.hstack(head_outputs))
        return torch.stack(result)


class MHA(nn.Module):
    """
    Standard Multi-Head Attention (MHA) module with support for masking and dropout.

    Args:
        d_model (int): Embedding dimension.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.att_scores: Tensor | None = None

    @staticmethod
    def attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
        dropout: nn.Dropout,
    ) -> Tuple[Tensor, Tensor]:
        """
        Scaled dot-product attention.

        Args:
            query (Tensor): (batch, heads, seq_len, d_k)
            key (Tensor): Same shape as query.
            value (Tensor): Same shape as query.
            mask (Optional[Tensor]): (batch, 1, 1, seq_len) or broadcastable
            dropout (nn.Dropout): Dropout module

        Returns:
            Tuple[Tensor, Tensor]: (attention output, attention weights)
        """
        d_k = query.shape[-1]
        att_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, float("-inf"))

        att_weights = att_scores.softmax(dim=-1)

        if dropout is not None:
            att_weights = dropout(att_weights)

        return att_weights @ value, att_weights

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
        """
        Forward pass through MHA block.

        Args:
            q, k, v (Tensor): Shape (batch, seq_len, d_model)
            mask (Optional[Tensor]): Attention mask

        Returns:
            Tensor: Shape (batch, seq_len, d_model)
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(
            query.shape[0], query.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(
            1, 2
        )
        value = value.view(
            value.shape[0], value.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)

        output, self.att_scores = self.attention(query, key, value, mask, self.dropout)

        output = (
            output.transpose(1, 2).contiguous().view(output.shape[0], -1, self.d_model)
        )
        return cast(Tensor, self.w_o(output))


class ShiftedWindowMSA(nn.Module):
    """
    Shifted Window Multi-Head Self-Attention for Vision Transformers.

    Args:
        embed_dim (int): Embedding dimension.
        n_heads (int): Number of attention heads.
        window_size (int): Size of local attention window.
        mask (bool): Whether to apply shift and masking (used in Swin).
    """

    def __init__(
        self, embed_dim: int, n_heads: int, window_size: int, mask: bool = False
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.window_size = window_size
        self.mask = mask
        self.embeddings = RelativeEmbedding()

        self.proj1 = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies shifted window attention.

        Args:
            x (Tensor): Input of shape (B, N, C) where N = H * W

        Returns:
            Tensor: Output tensor of shape (B, N, C)
        """
        device = x.device
        head_dim = self.embed_dim // self.n_heads
        height = width = int(math.sqrt(x.shape[1]))

        x = self.proj1(x)
        x = rearrange(x, "b (h w) (c k) -> b h w c k", h=height, w=width, k=3)

        if self.mask:
            x = torch.roll(
                x, shifts=(-self.window_size // 2, -self.window_size // 2), dims=(1, 2)
            )

        x = rearrange(
            x,
            "b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K",
            H=self.n_heads,
            m1=self.window_size,
            m2=self.window_size,
        )

        q, k, v = x.chunk(3, dim=6)
        q, k, v = q.squeeze(-1), k.squeeze(-1), v.squeeze(-1)

        attn_scores = (q @ k.transpose(4, 5)) / math.sqrt(head_dim)
        attn_scores = self.embeddings(attn_scores)

        if self.mask:
            mask_matrix = torch.zeros(
                (self.window_size**2, self.window_size**2), device=device
            )
            half = self.window_size * (self.window_size // 2)
            mask_matrix[-half:, :-half] = float("-inf")
            mask_matrix[:-half, -half:] = float("-inf")

            cross_mask = rearrange(
                mask_matrix,
                "(r w1) (c w2) -> (w1 r) (w2 c)",
                w1=self.window_size,
                w2=self.window_size,
            )

            attn_scores[:, :, -1, :] += mask_matrix
            attn_scores[:, :, :, -1] += cross_mask

        attention = functional.softmax(attn_scores, dim=-1)
        attended = attention @ v

        x = rearrange(
            attended,
            "b H h w (m1 m2) E -> b (h m1) (w m2) (H E)",
            m1=self.window_size,
            m2=self.window_size,
        )

        if self.mask:
            x = torch.roll(
                x, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2)
            )

        return cast(Tensor, self.proj2(rearrange(x, "b h w c -> b (h w) c")))
