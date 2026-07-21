from __future__ import annotations

import torch
from torch import nn


class TimeSeriesEncoder(nn.Module):
    """Small bidirectional GRU encoder with learned temporal-token queries."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        temporal_dim: int = 256,
        num_temporal_tokens: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        query_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_temporal_tokens = num_temporal_tokens
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_projection = nn.Linear(hidden_dim * 2, temporal_dim)
        # The queries are dotted against LayerNormed encoder outputs, whose norm is
        # sqrt(temporal_dim) (~16 for 256). Initialising them at 0.02 makes them ~50x
        # smaller, so after the sqrt(d) division the logits span ~0.2 while a softmax
        # over T steps needs a span of order log(T) (~6.9 for T=1000) to depart from
        # uniform. The pooling then degenerates into a plain temporal mean — and
        # gradients barely escape it, because a uniform softmax is a flat region.
        # ``query_init_std=1.0`` puts the queries on the same scale as the keys.
        self.token_queries = nn.Parameter(
            torch.randn(num_temporal_tokens, temporal_dim) * query_init_std
        )
        self.norm = nn.LayerNorm(temporal_dim)

    def _canonicalize(self, time_series: torch.Tensor) -> torch.Tensor:
        if time_series.ndim == 1:
            time_series = time_series[None, :, None]
        elif time_series.ndim == 2:
            if time_series.shape[-1] == self.input_dim:
                time_series = time_series[None, :, :]
            else:
                time_series = time_series[:, :, None]
        elif time_series.ndim != 3:
            raise ValueError("time_series must have shape [T], [T,V], or [B,T,V]")
        if time_series.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} variables, got {time_series.shape[-1]}. "
                "Set --input-dim to the dataset's padded feature count."
            )
        return time_series

    def forward(
        self, time_series: torch.Tensor, time_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._canonicalize(time_series).float()
        if time_mask is None:
            time_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            time_mask = time_mask.bool()

        # Per-variable normalization keeps scales stable while preserving temporal shape.
        valid = time_mask.unsqueeze(-1)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (x * valid).sum(dim=1, keepdim=True) / count
        variance = ((x - mean).square() * valid).sum(dim=1, keepdim=True) / count
        x = (x - mean) / torch.sqrt(variance + 1e-5)
        x = x.masked_fill(~valid, 0.0)

        encoded, _ = self.gru(self.input_projection(x))
        encoded = self.norm(self.output_projection(encoded))
        scores = torch.einsum("btd,kd->bkt", encoded, self.token_queries)
        scores = scores / (encoded.shape[-1] ** 0.5)
        scores = scores.masked_fill(~time_mask[:, None, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        temporal_tokens = torch.einsum("bkt,btd->bkd", attention, encoded)
        return temporal_tokens, attention
