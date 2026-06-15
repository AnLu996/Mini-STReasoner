from torch import nn


class TemporalProjector(nn.Module):
    def __init__(self, temporal_dim: int, llm_hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(temporal_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size),
        )

    def forward(self, temporal_tokens):
        return self.net(temporal_tokens)
