from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from ai_research_portfolio.lob_models.baselines import (
    BiN,
    CTABL,
    DAIN,
    DeepLOB,
    HLOBInspired,
)


def _assert_finite_gradients(model: nn.Module, loss: torch.Tensor) -> None:
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DeepLOB(feature_dim=8, num_classes=3, channels=4, hidden_size=6),
        lambda: CTABL(
            feature_dim=8,
            time_steps=12,
            num_classes=3,
            hidden_features=6,
            hidden_time=5,
            dropout=0.0,
        ),
        lambda: HLOBInspired(
            node_count=4,
            node_features=2,
            num_classes=3,
            graph_hidden=5,
            temporal_hidden=6,
        ),
    ],
)
def test_classifiers_have_common_shape_and_gradients(
    factory: Callable[[], nn.Module],
) -> None:
    torch.manual_seed(7)
    model = factory()
    x = torch.randn(2, 12, 8)

    logits = model(x)

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    _assert_finite_gradients(model, logits.square().mean())


def test_ctabl_attention_and_fixed_diagonal_invariants() -> None:
    torch.manual_seed(3)
    model = CTABL(
        feature_dim=8,
        time_steps=12,
        num_classes=3,
        hidden_features=7,
        hidden_time=6,
        dropout=0.0,
    )
    x = torch.randn(2, 12, 8)

    logits, attention = model(x, return_attention=True)
    mixer = model.temporal_attention.attention_matrix()

    assert logits.shape == (2, 3)
    assert attention.shape == (2, 3, 6)
    torch.testing.assert_close(
        attention.sum(dim=-1), torch.ones(2, 3), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        torch.diagonal(mixer), torch.full((6,), 1.0 / 6.0)
    )


def test_bin_normalises_both_axes_and_keeps_nonnegative_weights() -> None:
    torch.manual_seed(11)
    model = BiN(feature_dim=8, time_steps=12, eps=1e-8)
    x = torch.randn(3, 12, 8)

    across_time, across_features = model.components(x)
    output = model(x)

    assert output.shape == x.shape
    torch.testing.assert_close(
        across_time.mean(dim=1), torch.zeros(3, 8), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        across_time.std(dim=1, unbiased=False),
        torch.ones(3, 8),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        across_features.mean(dim=2),
        torch.zeros(3, 12),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        across_features.std(dim=2, unbiased=False),
        torch.ones(3, 12),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.all(model.mixture_weights >= 0)
    _assert_finite_gradients(model, output.square().mean())


def test_bin_is_finite_for_constant_windows() -> None:
    model = BiN(feature_dim=8, time_steps=12)
    output = model(torch.ones(2, 12, 8))
    assert torch.isfinite(output).all()


def test_dain_state_bounds_parameter_groups_and_gradients() -> None:
    torch.manual_seed(13)
    model = DAIN(feature_dim=8)
    x = torch.randn(2, 12, 8)

    output, state = model(x, return_state=True)

    assert output.shape == x.shape
    assert state["shift"].shape == (2, 8)
    assert state["scale"].shape == (2, 8)
    assert state["gate"].shape == (2, 8)
    assert torch.all(state["scale"] > 0)
    assert torch.all((state["gate"] >= 0) & (state["gate"] <= 1))

    groups = model.optimizer_groups(adaptation_lr=2e-4)
    grouped_parameters = [parameter for group in groups for parameter in group["params"]]
    assert [group["name"] for group in groups] == [
        "dain_shift",
        "dain_scale",
        "dain_gate",
    ]
    assert all(group["lr"] == pytest.approx(2e-4) for group in groups)
    assert len({id(parameter) for parameter in grouped_parameters}) == len(grouped_parameters)
    assert {id(parameter) for parameter in grouped_parameters} == {
        id(parameter) for parameter in model.parameters()
    }
    _assert_finite_gradients(model, output.square().mean())


def test_hlob_inspired_normalises_user_graph() -> None:
    adjacency = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    model = HLOBInspired(
        node_count=4,
        node_features=2,
        graph_hidden=4,
        temporal_hidden=5,
        adjacency=adjacency,
    )

    normalised = model.normalised_adjacency
    assert normalised.shape == (4, 4)
    assert torch.isfinite(normalised).all()
    assert torch.all(torch.diagonal(normalised) > 0)
    torch.testing.assert_close(normalised, normalised.T)


def test_hlob_inspired_rejects_directed_adjacency() -> None:
    directed = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        HLOBInspired(node_count=2, node_features=2, adjacency=directed)


@pytest.mark.parametrize(
    ("model", "bad_input", "message"),
    [
        (DeepLOB(feature_dim=8), torch.randn(2, 8), "shape"),
        (CTABL(feature_dim=8, time_steps=12), torch.randn(2, 11, 8), "time_steps"),
        (DAIN(feature_dim=8), torch.randn(2, 12, 7), "feature_dim"),
    ],
)
def test_models_reject_incompatible_inputs(
    model: nn.Module, bad_input: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        model(bad_input)
