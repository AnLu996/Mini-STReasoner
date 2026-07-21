import torch
from torch import nn


class TemporalProjector(nn.Module):
    """Project temporal tokens into the LLM embedding space.

    The trailing LayerNorm leaves every token with norm ``sqrt(llm_hidden_size)``
    times its gain. With the default gain of 1 that is 32 for a 1024-wide model,
    while an LLM's own input embeddings have norm close to 1 — so the temporal
    tokens arrive more than an order of magnitude out of scale and behave like a
    large near-constant prefix rather than a channel of information.

    ``output_scale`` initialises the gain so the projected tokens match the norm
    of the text embeddings they are concatenated with. It stays trainable, so the
    model can still move away from that starting point.
    """

    def __init__(
        self, temporal_dim: int, llm_hidden_size: int, output_scale: float | None = None
    ) -> None:
        super().__init__()
        # Kept as a single Sequential so the parameter names stay ``net.*`` and
        # checkpoints written before ``output_scale`` existed still load.
        self.net = nn.Sequential(
            nn.Linear(temporal_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size),
        )
        self.output_scale = output_scale
        if output_scale is not None:
            with torch.no_grad():
                self.norm.weight.fill_(output_scale)

    @property
    def norm(self) -> nn.LayerNorm:
        return self.net[-1]

    def forward(self, temporal_tokens):
        return self.net(temporal_tokens)
