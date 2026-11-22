"""Model definitions. All models take (batch, channels, time) and return logits.

I deliberately kept these small (tens of thousands to a couple hundred
thousand parameters). The point of the project is something that can run
on a phone in real time, so a 3M-parameter model is not interesting here.
"""

from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    """Plain 3-block 1D CNN with global average pooling."""

    def __init__(self, in_ch: int, n_classes: int, width: int = 64, dropout: float = 0.3):
        super().__init__()

        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())

        self.features = nn.Sequential(
            block(in_ch, width, 7), nn.MaxPool1d(2),
            block(width, width * 2, 5), nn.MaxPool1d(2),
            block(width * 2, width * 2, 3),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 2, n_classes))

    def forward(self, x):
        return self.head(self.features(x).mean(-1))


class ConvGRU(nn.Module):
    """DeepConvLSTM-style: conv feature extractor followed by a recurrent layer.

    Ordóñez & Roggen (2016) use LSTMs; a GRU gave the same accuracy here
    with fewer parameters.
    """

    def __init__(self, in_ch: int, n_classes: int, width: int = 64, hidden: int = 96,
                 dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(width, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        h = self.conv(x).transpose(1, 2)  # (B, T, F)
        _, last = self.gru(h)
        return self.head(last[-1])


class _TCNBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.ConstantPad1d((pad, 0), 0.0), nn.Conv1d(ch, ch, k, dilation=dilation),
            nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout),
            nn.ConstantPad1d((pad, 0), 0.0), nn.Conv1d(ch, ch, k, dilation=dilation),
            nn.BatchNorm1d(ch), nn.ReLU(),
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class TCN(nn.Module):
    """Causal dilated TCN (Bai et al., 2018). Receptive field with the
    defaults is 1 + 2*(k-1)*(1+2+4+8) = 181 samples, i.e. covers a 128-sample window."""

    def __init__(self, in_ch: int, n_classes: int, width: int = 64, levels: int = 4,
                 kernel: int = 4, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, width, 1)
        self.blocks = nn.Sequential(*[_TCNBlock(width, kernel, 2 ** i, dropout) for i in range(levels)])
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        return self.head(self.blocks(self.inp(x)).mean(-1))


class PatchTransformer(nn.Module):
    """Small transformer over non-overlapping patches of the window
    (the patching idea from PatchTST, but channel-mixing instead of
    channel-independent, which worked better for classification here)."""

    def __init__(self, in_ch: int, n_classes: int, patch: int = 8, dim: int = 64,
                 depth: int = 3, heads: int = 4, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv1d(in_ch, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, max_len // patch, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        z = self.embed(x).transpose(1, 2)  # (B, n_patches, dim)
        z = z + self.pos[:, : z.shape[1]]
        z = self.encoder(z)
        return self.head(self.norm(z.mean(1)))


MODELS = {
    "cnn": CNN1D,
    "convgru": ConvGRU,
    "tcn": TCN,
    "transformer": PatchTransformer,
}


def build_model(name: str, in_ch: int, n_classes: int, **kw) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, choose from {list(MODELS)}")
    return MODELS[name](in_ch, n_classes, **kw)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def receptive_field_ok(model: nn.Module, win: int) -> bool:
    if isinstance(model, PatchTransformer):
        return win % model.patch == 0 and win // model.patch <= model.pos.shape[1]
    return True


class WithNormalization(nn.Module):
    """Wraps a model so it takes raw sensor values.

    Keeping the scaler inside the network means the exported ONNX file is
    self-contained: the phone just feeds it the raw window.
    """

    def __init__(self, net: nn.Module, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


__all__ = ["build_model", "n_params", "MODELS", "CNN1D", "ConvGRU", "TCN", "PatchTransformer",
           "WithNormalization",
           "receptive_field_ok"]
