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

    ``kind`` selecciona la arquitectura del proyector (F05.1 del plan):

    - ``"mlp"`` (por defecto, usado en E1/E2): la MLP compacta de siempre —
      Linear → GELU → Linear → LayerNorm. Un único bloque residual implícito.
    - ``"expressive"`` (usado en E3): añade una capa oculta intermedia con
      residual y dropout. Responde a la pregunta "¿el cuello es el proyector?"
      del brief: si al pasar de ``mlp`` a ``expressive`` la cascada de sondas
      sube tras ``h_proj``, la firma es H3 y el remedio es un proyector más
      expresivo. Si no sube, el cuello está aguas arriba (H2) o aguas abajo
      (H4/H5).
    """

    def __init__(
        self,
        temporal_dim: int,
        llm_hidden_size: int,
        output_scale: float | None = None,
        kind: str = "mlp",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            # Kept as a single Sequential so the parameter names stay ``net.*`` and
            # checkpoints written before ``kind`` existed still load.
            self.net = nn.Sequential(
                nn.Linear(temporal_dim, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
                nn.LayerNorm(llm_hidden_size),
            )
        elif kind == "expressive":
            # Q-Former ligero: dos capas ocultas, residual, dropout. Duplica los
            # parámetros del proyector (todavía ínfimos frente al LLM) y da al
            # modelo margen para re-linealizar lo que el pooling comprimió.
            self.net = nn.Sequential(
                nn.Linear(temporal_dim, llm_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                _ResidualMLP(llm_hidden_size, dropout=dropout),
                _ResidualMLP(llm_hidden_size, dropout=dropout),
                nn.LayerNorm(llm_hidden_size),
            )
        else:
            raise ValueError(f"kind desconocido: {kind!r}. Usa 'mlp' o 'expressive'.")
        self.output_scale = output_scale
        if output_scale is not None:
            with torch.no_grad():
                self.norm.weight.fill_(output_scale)

    @property
    def norm(self) -> nn.LayerNorm:
        return self.net[-1]

    def forward(self, temporal_tokens):
        return self.net(temporal_tokens)


class _ResidualMLP(nn.Module):
    """Bloque residual con LayerNorm-then-Linear-GELU-Linear, dropout al final."""

    def __init__(self, hidden: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        return x + self.drop(h)
