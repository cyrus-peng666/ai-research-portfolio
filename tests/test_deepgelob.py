import numpy as np
import pytest
import torch

from ai_research_portfolio.lob_models.deepgelob import DeepGELOB
from ai_research_portfolio.lob_models.evaluation import (
    cost_sensitivity,
    regression_metrics,
    time_thin_mask,
)
from ai_research_portfolio.lob_models.labels import (
    TrainFittedStandardizer,
    build_day_aware_windows,
    chronological_day_split,
    day_aware_window_end_indices,
    past_only_standardize_by_day,
    strict_future_average_return,
)
from ai_research_portfolio.lob_models.synthetic import generate_synthetic_lob


def test_deepgelob_shape_and_gradient() -> None:
    torch.manual_seed(1)
    model = DeepGELOB(levels=5, dropout=0.0)
    book = torch.randn(3, 12, 2, 5, 4)
    delta = torch.rand(3, 12)
    prediction = model(book, delta)
    assert prediction.shape == (3,)
    prediction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_deepgelob_rejects_invalid_semantic_shape() -> None:
    model = DeepGELOB(levels=5)
    with pytest.raises(ValueError, match="book must have shape"):
        model(torch.randn(2, 8, 2, 4, 4), torch.rand(2, 8))
    with pytest.raises(ValueError, match="nonnegative"):
        model(torch.randn(2, 8, 2, 5, 4), -torch.ones(2, 8))


def test_strict_target_uses_future_only_and_never_crosses_day() -> None:
    mid = np.array([100, 101, 102, 103, 104, 200, 202, 204, 206, 208], dtype=float)
    timestamps = np.tile(np.arange(5, dtype=float), 2)
    days = np.repeat([0, 1], 5)
    target = strict_future_average_return(mid, timestamps, days, horizon_seconds=2.0)
    assert target[0] == pytest.approx(101.5 / 100.0 - 1.0)
    assert target[1] == pytest.approx(102.5 / 101.0 - 1.0)
    assert np.isnan(target[3:5]).all()
    assert target[5] == pytest.approx(203.0 / 200.0 - 1.0)


def test_day_aware_windows_and_chronological_split() -> None:
    days = np.repeat(np.arange(4), 4)
    ends = day_aware_window_end_indices(days, 3)
    assert ends.tolist() == [2, 3, 6, 7, 10, 11, 14, 15]
    split = chronological_day_split(days, train_fraction=0.5, validation_fraction=0.25)
    assert split.train_days == (0, 1)
    assert split.validation_days == (2,)
    assert split.test_days == (3,)
    with pytest.raises(ValueError, match="contiguous"):
        chronological_day_split([0, 1, 0])


def test_normalization_never_fits_on_test_or_current_day() -> None:
    values = np.array([[0.0], [2.0], [100.0], [102.0]])
    fitted = TrainFittedStandardizer().fit_transform(values, [True, True, False, False])
    assert fitted[:2].mean() == pytest.approx(0.0)
    assert fitted[2, 0] == pytest.approx(99.0)

    by_day = past_only_standardize_by_day(values, [0, 0, 1, 1], lookback_days=1)
    assert np.isnan(by_day[:2]).all()
    assert by_day[2, 0] == pytest.approx(99.0)


def test_thinning_resets_by_day_and_cost_is_monotone() -> None:
    times = np.array([0.0, 0.3, 1.1, 0.0, 0.2, 1.2])
    days = np.repeat([0, 1], 3)
    selected = time_thin_mask(times, days, 1.0)
    assert selected.tolist() == [True, False, True, True, False, True]

    true = np.full(6, 0.001)
    pred = np.full(6, 0.002)
    report = cost_sensitivity(
        true,
        pred,
        times,
        days,
        activation_threshold=0.001,
        cooldown_seconds=1.0,
        round_trip_costs_bps=(0, 2),
    )
    assert report["selected_signals"] == 4
    assert report["cost_scenarios_bps"]["0"]["mean_net_bps"] == pytest.approx(10.0)
    assert report["cost_scenarios_bps"]["2"]["mean_net_bps"] == pytest.approx(8.0)


def test_synthetic_pipeline_has_valid_shapes_and_metrics() -> None:
    stream = generate_synthetic_lob(days=3, events_per_day=32, levels=4, seed=2)
    target = strict_future_average_return(
        stream.mid_price, stream.timestamps, stream.day_ids, horizon_seconds=0.5
    )
    windowed = build_day_aware_windows(
        stream.book, stream.time_delta, target, stream.day_ids, window_size=8
    )
    assert windowed.book.ndim == 5
    assert windowed.book.shape[1:] == (8, 2, 4, 4)
    assert np.isfinite(windowed.target).all()
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert metrics["samples"] == 3
    assert metrics["spearman_ic"] == pytest.approx(1.0)
