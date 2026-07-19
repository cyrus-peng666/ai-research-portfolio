"""Compact, clean-room architecture studies for limit-order-book models.

All modules use one public input convention: ``(batch, time, features)``.
They return unnormalised class logits unless the class is an input-normalisation
layer (``BiN`` and ``DAIN``), in which case the input shape is preserved.

These implementations capture the central architectural ideas of the cited
model families. They are not copies of the authors' code and have not been
validated against the papers' complete data or training protocols.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_lob_input(
    x: Tensor,
    *,
    feature_dim: int,
    time_steps: int | None = None,
) -> None:
    if x.ndim != 3:
        raise ValueError(
            "LOB input must have shape (batch, time, features); "
            f"received {tuple(x.shape)}"
        )
    if x.shape[-1] != feature_dim:
        raise ValueError(
            f"expected feature_dim={feature_dim}, received {x.shape[-1]}"
        )
    if time_steps is not None and x.shape[1] != time_steps:
        raise ValueError(
            f"expected time_steps={time_steps}, received {x.shape[1]}"
        )
    if not torch.is_floating_point(x):
        raise TypeError("LOB input must be a floating-point tensor")


class DeepLOB(nn.Module):
    """DeepLOB-style local feature extraction, multi-scale time mixing and LSTM.

    This compact implementation preserves the model family's main inductive
    biases: shared convolutions compress paired LOB fields, multiple temporal
    kernels detect patterns at different scales, and an LSTM reads the ordered
    sequence. It is intentionally smaller than common paper implementations.
    """

    def __init__(
        self,
        feature_dim: int = 40,
        num_classes: int = 3,
        channels: int = 16,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        if feature_dim < 4 or feature_dim % 4 != 0:
            raise ValueError("feature_dim must be a positive multiple of four")

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)

        self.feature_encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(3, 2),
                stride=(1, 2),
                padding=(1, 0),
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.temporal_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                )
                for k in (1, 3, 5)
            ]
        )
        self.temporal_model = nn.LSTM(
            input_size=3 * channels,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        _validate_lob_input(x, feature_dim=self.feature_dim)

        encoded = self.feature_encoder(x.unsqueeze(1))
        encoded = encoded.mean(dim=-1)  # (batch, channels, time)
        multi_scale = torch.cat(
            [branch(encoded) for branch in self.temporal_branches], dim=1
        )
        sequence = multi_scale.transpose(1, 2)
        output, _ = self.temporal_model(sequence)
        return self.classifier(output[:, -1])


class _BilinearProjection(nn.Module):
    """Learn a separable feature-axis and time-axis projection."""

    def __init__(
        self,
        in_features: int,
        in_time: int,
        out_features: int,
        out_time: int,
    ) -> None:
        super().__init__()
        self.feature_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.time_weight = nn.Parameter(torch.empty(in_time, out_time))
        self.bias = nn.Parameter(torch.zeros(out_features, out_time))
        nn.init.xavier_uniform_(self.feature_weight)
        nn.init.xavier_uniform_(self.time_weight)

    def forward(self, x: Tensor) -> Tensor:
        feature_mapped = torch.einsum("of,bft->bot", self.feature_weight, x)
        return torch.einsum(
            "bot,tu->bou", feature_mapped, self.time_weight
        ) + self.bias.unsqueeze(0)


class _TemporalAttentionBilinear(nn.Module):
    """Final bilinear projection with a row-wise temporal attention mask."""

    def __init__(
        self,
        in_features: int,
        in_time: int,
        out_features: int,
        out_time: int = 1,
    ) -> None:
        super().__init__()
        self.in_time = int(in_time)
        self.feature_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.raw_attention_matrix = nn.Parameter(torch.full((in_time, in_time), 1.0 / in_time))
        self.time_weight = nn.Parameter(torch.empty(in_time, out_time))
        self.bias = nn.Parameter(torch.zeros(out_features, out_time))
        self.attention_mix_logit = nn.Parameter(torch.tensor(0.0))
        nn.init.xavier_uniform_(self.feature_weight)
        nn.init.xavier_uniform_(self.time_weight)

    def attention_matrix(self) -> Tensor:
        """Return the learned time mixer with a fixed ``1 / T`` diagonal."""

        eye = torch.eye(
            self.in_time,
            dtype=self.raw_attention_matrix.dtype,
            device=self.raw_attention_matrix.device,
        )
        return self.raw_attention_matrix * (1.0 - eye) + eye / self.in_time

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feature_mapped = torch.einsum("of,bft->bot", self.feature_weight, x)
        energy = torch.matmul(feature_mapped, self.attention_matrix())
        attention = torch.softmax(energy, dim=-1)

        residual_weight = torch.sigmoid(self.attention_mix_logit)
        attended = feature_mapped * (
            residual_weight + (1.0 - residual_weight) * attention
        )
        output = torch.einsum(
            "bot,tu->bou", attended, self.time_weight
        ) + self.bias.unsqueeze(0)
        return output, attention


class CTABL(nn.Module):
    """Compact C(TABL)-style classifier with separable bilinear mappings."""

    def __init__(
        self,
        feature_dim: int = 40,
        time_steps: int = 100,
        num_classes: int = 3,
        hidden_features: int = 32,
        hidden_time: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if time_steps < 2:
            raise ValueError("time_steps must be at least two")
        hidden_time = hidden_time or max(4, time_steps // 2)

        self.feature_dim = int(feature_dim)
        self.time_steps = int(time_steps)
        self.hidden_time = int(hidden_time)
        self.bilinear = _BilinearProjection(
            in_features=feature_dim,
            in_time=time_steps,
            out_features=hidden_features,
            out_time=hidden_time,
        )
        self.dropout = nn.Dropout(dropout)
        self.temporal_attention = _TemporalAttentionBilinear(
            in_features=hidden_features,
            in_time=hidden_time,
            out_features=num_classes,
            out_time=1,
        )

    def forward(
        self, x: Tensor, *, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        _validate_lob_input(
            x,
            feature_dim=self.feature_dim,
            time_steps=self.time_steps,
        )
        matrix = x.transpose(1, 2)  # (batch, features, time)
        hidden = self.dropout(F.gelu(self.bilinear(matrix)))
        output, attention = self.temporal_attention(hidden)
        logits = output.squeeze(-1)
        if return_attention:
            return logits, attention
        return logits


class BiN(nn.Module):
    """Bilinear input normalisation along both time and feature axes.

    The first branch standardises every feature over time. The second branch
    standardises every time step over features. Learned, non-negative weights
    combine the two branches without forcing them to sum to one.
    """

    def __init__(
        self,
        feature_dim: int = 40,
        time_steps: int = 100,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.time_steps = int(time_steps)
        self.eps = float(eps)

        self.time_axis_scale = nn.Parameter(torch.ones(feature_dim))
        self.time_axis_shift = nn.Parameter(torch.zeros(feature_dim))
        self.feature_axis_scale = nn.Parameter(torch.ones(time_steps))
        self.feature_axis_shift = nn.Parameter(torch.zeros(time_steps))
        initial_raw = math.log(math.expm1(0.5))
        self.raw_mixture_weights = nn.Parameter(torch.full((2,), initial_raw))

    @property
    def mixture_weights(self) -> Tensor:
        return F.softplus(self.raw_mixture_weights)

    def components(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the two affine-normalised branches before mixing."""

        _validate_lob_input(
            x,
            feature_dim=self.feature_dim,
            time_steps=self.time_steps,
        )

        time_mean = x.mean(dim=1, keepdim=True)
        time_var = x.var(dim=1, keepdim=True, unbiased=False)
        across_time = (x - time_mean) / torch.sqrt(time_var + self.eps)
        across_time = (
            across_time * self.time_axis_scale.view(1, 1, -1)
            + self.time_axis_shift.view(1, 1, -1)
        )

        feature_mean = x.mean(dim=2, keepdim=True)
        feature_var = x.var(dim=2, keepdim=True, unbiased=False)
        across_features = (x - feature_mean) / torch.sqrt(feature_var + self.eps)
        across_features = (
            across_features * self.feature_axis_scale.view(1, -1, 1)
            + self.feature_axis_shift.view(1, -1, 1)
        )
        return across_time, across_features

    def forward(self, x: Tensor) -> Tensor:
        across_time, across_features = self.components(x)
        weights = self.mixture_weights
        return weights[0] * across_time + weights[1] * across_features


class DAIN(nn.Module):
    """Deep adaptive input normalisation with shift, scale and gate stages."""

    def __init__(self, feature_dim: int = 40, eps: float = 1e-5) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)
        self.shift_layer = nn.Linear(feature_dim, feature_dim)
        self.scale_layer = nn.Linear(feature_dim, feature_dim)
        self.gate_layer = nn.Linear(feature_dim, feature_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.shift_layer, self.scale_layer, self.gate_layer):
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def optimizer_groups(self, adaptation_lr: float) -> list[dict[str, object]]:
        """Return separate optimiser groups for the three adaptive stages."""

        if adaptation_lr <= 0:
            raise ValueError("adaptation_lr must be positive")
        return [
            {
                "name": "dain_shift",
                "params": list(self.shift_layer.parameters()),
                "lr": adaptation_lr,
            },
            {
                "name": "dain_scale",
                "params": list(self.scale_layer.parameters()),
                "lr": adaptation_lr,
            },
            {
                "name": "dain_gate",
                "params": list(self.gate_layer.parameters()),
                "lr": adaptation_lr,
            },
        ]

    def forward(
        self, x: Tensor, *, return_state: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        _validate_lob_input(x, feature_dim=self.feature_dim)

        summary = x.mean(dim=1)
        shift = self.shift_layer(summary)
        centered = x - shift.unsqueeze(1)

        root_mean_square = torch.sqrt(centered.square().mean(dim=1) + self.eps)
        scale = F.softplus(self.scale_layer(root_mean_square)) + self.eps
        scaled = centered / scale.unsqueeze(1)

        gate = torch.sigmoid(self.gate_layer(scaled.mean(dim=1)))
        output = scaled * gate.unsqueeze(1)
        if return_state:
            return output, {"shift": shift, "scale": scale, "gate": gate}
        return output


def _default_chain_adjacency(node_count: int) -> Tensor:
    adjacency = torch.zeros(node_count, node_count)
    if node_count > 1:
        index = torch.arange(node_count - 1)
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    return adjacency


def _normalise_adjacency(adjacency: Tensor) -> Tensor:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not torch.is_floating_point(adjacency):
        adjacency = adjacency.float()
    if not torch.isfinite(adjacency).all():
        raise ValueError("adjacency must contain only finite values")
    if (adjacency < 0).any():
        raise ValueError("adjacency must be non-negative")
    if not torch.allclose(adjacency, adjacency.T, atol=1e-6, rtol=0.0):
        raise ValueError("adjacency must be symmetric")

    with_self_loops = adjacency + torch.eye(
        adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device
    )
    degree = with_self_loops.sum(dim=1).clamp_min(1e-12)
    inv_sqrt_degree = degree.rsqrt()
    return (
        inv_sqrt_degree[:, None]
        * with_self_loops
        * inv_sqrt_degree[None, :]
    )


class HLOBInspired(nn.Module):
    """Graph-aware LOB classifier inspired by HLOB's relational premise.

    This is deliberately *not* an HLOB reproduction. It performs message
    passing over a user-provided, fixed node graph, pools node embeddings and
    models their temporal evolution with an LSTM. Graph estimation, TMFG and
    higher-order simplex convolutions are outside this module's scope.

    Flattened features must follow node-major order:
    ``node_0_feature_0, node_0_feature_1, ..., node_n_feature_c``.
    """

    def __init__(
        self,
        node_count: int = 20,
        node_features: int = 2,
        num_classes: int = 3,
        graph_hidden: int = 16,
        temporal_hidden: int = 32,
        graph_layers: int = 2,
        adjacency: Tensor | Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__()
        if node_count < 2:
            raise ValueError("node_count must be at least two")
        if node_features < 1:
            raise ValueError("node_features must be positive")
        if graph_layers < 1:
            raise ValueError("graph_layers must be positive")

        self.node_count = int(node_count)
        self.node_features = int(node_features)
        self.feature_dim = self.node_count * self.node_features

        if adjacency is None:
            adjacency_tensor = _default_chain_adjacency(self.node_count)
        else:
            adjacency_tensor = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency_tensor.shape != (self.node_count, self.node_count):
            raise ValueError(
                "adjacency shape must equal "
                f"({self.node_count}, {self.node_count})"
            )
        self.register_buffer(
            "normalised_adjacency", _normalise_adjacency(adjacency_tensor)
        )

        self.node_encoder = nn.Linear(self.node_features, graph_hidden)
        self.graph_updates = nn.ModuleList(
            [nn.Linear(graph_hidden, graph_hidden) for _ in range(graph_layers)]
        )
        self.graph_norms = nn.ModuleList(
            [nn.LayerNorm(graph_hidden) for _ in range(graph_layers)]
        )
        self.temporal_model = nn.LSTM(
            input_size=2 * graph_hidden,
            hidden_size=temporal_hidden,
            batch_first=True,
        )
        self.classifier = nn.Linear(temporal_hidden, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        _validate_lob_input(x, feature_dim=self.feature_dim)
        batch, time, _ = x.shape
        nodes = x.reshape(batch, time, self.node_count, self.node_features)
        hidden = F.gelu(self.node_encoder(nodes))

        for update, norm in zip(self.graph_updates, self.graph_norms, strict=True):
            message = torch.einsum(
                "ij,btjh->btih", self.normalised_adjacency, hidden
            )
            hidden = norm(hidden + F.gelu(update(message)))

        pooled = torch.cat(
            [hidden.mean(dim=2), hidden.amax(dim=2)], dim=-1
        )
        sequence, _ = self.temporal_model(pooled)
        return self.classifier(sequence[:, -1])


__all__ = ["BiN", "CTABL", "DAIN", "DeepLOB", "HLOBInspired"]
