from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .temporal_projector import TemporalProjector
from .time_series_encoder import TimeSeriesEncoder


@dataclass
class MiniSTReasonerOutput:
    llm_output: Any
    temporal_attention: torch.Tensor


class MiniSTReasoner(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        input_dim: int = 1,
        temporal_hidden_dim: int = 128,
        temporal_dim: int = 256,
        num_temporal_tokens: int = 4,
        temporal_num_layers: int = 1,
        query_init_std: float = 0.02,
        match_embedding_scale: bool = False,
    ) -> None:
        super().__init__()
        self.llm = llm
        hidden_size = llm.config.hidden_size
        self.time_series_encoder = TimeSeriesEncoder(
            input_dim=input_dim,
            hidden_dim=temporal_hidden_dim,
            temporal_dim=temporal_dim,
            num_temporal_tokens=num_temporal_tokens,
            num_layers=temporal_num_layers,
            query_init_std=query_init_std,
        )
        output_scale = (
            self.embedding_norm(llm) / hidden_size**0.5 if match_embedding_scale else None
        )
        self.temporal_projector = TemporalProjector(temporal_dim, hidden_size, output_scale)
        self.num_temporal_tokens = num_temporal_tokens

    @staticmethod
    def embedding_norm(llm: nn.Module) -> float:
        """Mean L2 norm of the LLM's input embeddings, the scale to match."""
        weight = llm.get_input_embeddings().weight
        return float(weight.detach().float().norm(dim=-1).mean())

    @property
    def input_device(self) -> torch.device:
        return self.llm.get_input_embeddings().weight.device

    def encode_modalities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        time_series: torch.Tensor,
        time_mask: torch.Tensor | None = None,
        use_text: bool = True,
        use_series: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.input_device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        if not use_text:
            text_embeds = text_embeds[:, :0]
            attention_mask = attention_mask[:, :0]

        if use_series:
            encoder_device = next(self.time_series_encoder.parameters()).device
            temporal_tokens, temporal_attention = self.time_series_encoder(
                time_series.to(encoder_device),
                time_mask.to(encoder_device) if time_mask is not None else None,
            )
            temporal_embeds = self.temporal_projector(temporal_tokens).to(device=device, dtype=text_embeds.dtype)
            temporal_mask = torch.ones(
                temporal_embeds.shape[:2], dtype=attention_mask.dtype, device=device
            )
        else:
            batch = input_ids.shape[0]
            steps = time_series.shape[-2] if time_series.ndim >= 2 else time_series.shape[0]
            temporal_attention = torch.zeros(
                batch, 0, steps, device=next(self.time_series_encoder.parameters()).device
            )
            temporal_embeds = text_embeds[:, :0]
            temporal_mask = attention_mask[:, :0]

        inputs_embeds = torch.cat([temporal_embeds, text_embeds], dim=1)
        combined_mask = torch.cat([temporal_mask, attention_mask], dim=1)
        return inputs_embeds, combined_mask, temporal_attention

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        time_series: torch.Tensor,
        time_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> MiniSTReasonerOutput:
        inputs_embeds, combined_mask, temporal_attention = self.encode_modalities(
            input_ids, attention_mask, time_series, time_mask
        )
        combined_labels = None
        if labels is not None:
            ignored = torch.full(
                (labels.shape[0], self.num_temporal_tokens),
                -100,
                dtype=labels.dtype,
                device=self.input_device,
            )
            combined_labels = torch.cat([ignored, labels.to(self.input_device)], dim=1)
        output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
            **kwargs,
        )
        return MiniSTReasonerOutput(output, temporal_attention)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        time_series: torch.Tensor,
        time_mask: torch.Tensor | None = None,
        use_text: bool = True,
        use_series: bool = True,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds, combined_mask, temporal_attention = self.encode_modalities(
            input_ids,
            attention_mask,
            time_series,
            time_mask,
            use_text=use_text,
            use_series=use_series,
        )
        generated = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            **generation_kwargs,
        )
        return generated, temporal_attention
