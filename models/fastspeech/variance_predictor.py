from torch import nn, Tensor

from typing import TypedDict, Optional

from models.transformer.layer_norm import LayerNorm


class VariancePredictorConfig(TypedDict):
    filter_size: int
    kernel_size: int
    dropout: float


class VariancePredictor(nn.Module):
    def __init__(
        self,
        config: VariancePredictorConfig,
        in_channels: int,
        gin_channels: int = 0,
    ):
        super(VariancePredictor, self).__init__()
        self.conv = nn.ModuleList()
        self.filter_size = config["filter_size"]
        self.kernel_size = config["kernel_size"]
        self.dropout = config["dropout"]

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        for idx in range(2):
            in_channels = in_channels if idx == 0 else self.filter_size
            self.conv += [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        self.filter_size,
                        self.kernel_size,
                        stride=1,
                        padding=(self.kernel_size - 1) // 2,
                    ),
                    nn.ReLU(),
                    LayerNorm(self.filter_size, dim=1),
                    nn.Dropout(self.dropout),
                )
            ]
        self.linear = nn.Linear(self.filter_size, 1)

    def forward(self, xs: Tensor, x_masks: Optional[Tensor] = None, g: Optional[Tensor] = None) -> Tensor:
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """

        xs = xs.transpose(1, -1)  # (B, idim, Tmax)

        if g is not None:
            g = g.detach()
            xs = xs + self.cond(g)

        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        xs = self.linear(xs.transpose(1, 2)).transpose(1, 2)

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs
