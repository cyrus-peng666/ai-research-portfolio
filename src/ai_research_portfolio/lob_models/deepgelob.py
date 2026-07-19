"""Independent, public DeepGELOB core.

The model deliberately accepts an explicit semantic tensor instead of any
vendor-specific market-data schema.  The last axis is ordered as
``[price, volume, order_count, queue_age]`` and TimeDiff is supplied through
its own input.  This file is a clean implementation for the public portfolio;
it is not copied from an internship codebase.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class _SemanticBookBranch(nn.Module):
    """Encode one semantic view of both book sides at every event."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim < 4:
            raise ValueError("output_dim must be at least 4")

        self.encoder = nn.Sequential(
            nn.Conv2d(2, output_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(1, output_dim),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(1, output_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, view: Tensor) -> Tensor:
        # view: (batch, time, side=2, level, semantic_channel=2)
        batch, steps, sides, levels, channels = view.shape
        if sides != 2 or channels != 2:
            raise ValueError("semantic view must have shape (B, T, 2, L, 2)")

        event_images = view.permute(0, 1, 4, 2, 3).reshape(
            batch * steps, channels, sides, levels
        )
        encoded = self.encoder(event_images).flatten(1)
        return encoded.reshape(batch, steps, -1)


class DeepGELOB(nn.Module):
    """Semantically grouped LOB encoder followed by a recurrent predictor.

    Parameters
    ----------
    levels:
        Number of displayed levels on each side of the book.
    branch_dim:
        Width of the independent PV and NA branches.
    timediff_dim:
        Width of the event-time branch.
    hidden_dim:
        Hidden size of the GRU used to aggregate the historical window.

    Notes
    -----
    ``book`` must have shape ``(B, T, 2, levels, 4)``.  The two sides may use
    any consistent order, but the semantic channel order is fixed to price,
    volume, order count, and queue age.  ``time_delta`` is raw nonnegative
    elapsed time (or ``log1p`` elapsed time), shaped ``(B, T)`` or
    ``(B, T, 1)``.
    """

    feature_names = ("price", "volume", "order_count", "queue_age")

    def __init__(
        self,
        levels: int = 10,
        branch_dim: int = 24,
        timediff_dim: int = 8,
        hidden_dim: int = 48,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        if timediff_dim < 1 or hidden_dim < 1:
            raise ValueError("timediff_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.levels = int(levels)
        self.pv_branch = _SemanticBookBranch(branch_dim)
        self.na_branch = _SemanticBookBranch(branch_dim)
        self.timediff_branch = nn.Sequential(
            nn.Conv1d(1, timediff_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(timediff_dim, timediff_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.temporal = nn.GRU(
            input_size=2 * branch_dim + timediff_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _validate_inputs(self, book: Tensor, time_delta: Tensor) -> Tensor:
        if book.ndim != 5:
            raise ValueError("book must have shape (B, T, 2, levels, 4)")
        if book.shape[2] != 2 or book.shape[3] != self.levels or book.shape[4] != 4:
            raise ValueError(
                f"book must have shape (B, T, 2, {self.levels}, 4); got {tuple(book.shape)}"
            )
        if not torch.is_floating_point(book) or not torch.isfinite(book).all():
            raise ValueError("book must be a finite floating-point tensor")

        if time_delta.ndim == 3 and time_delta.shape[-1] == 1:
            time_delta = time_delta.squeeze(-1)
        if time_delta.ndim != 2 or time_delta.shape != book.shape[:2]:
            raise ValueError("time_delta must have shape (B, T) or (B, T, 1)")
        if not torch.is_floating_point(time_delta) or not torch.isfinite(time_delta).all():
            raise ValueError("time_delta must be a finite floating-point tensor")
        if torch.any(time_delta < 0):
            raise ValueError("time_delta must be nonnegative")
        return time_delta

    def forward(
        self,
        book: Tensor,
        time_delta: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """Return one continuous prediction per historical input window."""

        time_delta = self._validate_inputs(book, time_delta)
        pv = self.pv_branch(book[..., :2])
        na = self.na_branch(book[..., 2:4])
        dt = self.timediff_branch(time_delta.unsqueeze(1)).transpose(1, 2)
        sequence = torch.cat((pv, na, dt), dim=-1)
        output, _ = self.temporal(sequence)

        if lengths is None:
            final_state = output[:, -1]
        else:
            if lengths.ndim != 1 or lengths.shape[0] != book.shape[0]:
                raise ValueError("lengths must have shape (B,)")
            lengths = lengths.to(device=book.device, dtype=torch.long)
            if torch.any(lengths < 1) or torch.any(lengths > book.shape[1]):
                raise ValueError("each sequence length must be between 1 and T")
            row = torch.arange(book.shape[0], device=book.device)
            final_state = output[row, lengths - 1]

        return self.head(final_state).squeeze(-1)

