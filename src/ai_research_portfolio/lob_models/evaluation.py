"""Decision-aware evaluation for dense event-level predictions."""

from __future__ import annotations

from typing import Hashable, Iterable, Sequence

import numpy as np

from .labels import validate_day_sequence


def _aligned_finite(y_true: Sequence[float], y_pred: Sequence[float]):
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.ndim != 1 or pred.ndim != 1 or true.shape != pred.shape:
        raise ValueError("y_true and y_pred must be aligned one-dimensional arrays")
    mask = np.isfinite(true) & np.isfinite(pred)
    return true, pred, mask


def _average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float | int]:
    """Return dense regression metrics after a shared finite-value filter."""

    true, pred, mask = _aligned_finite(y_true, y_pred)
    true = true[mask]
    pred = pred[mask]
    if true.size == 0:
        return {
            "samples": 0,
            "pearson_ic": float("nan"),
            "spearman_ic": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "directional_accuracy": float("nan"),
        }
    return {
        "samples": int(true.size),
        "pearson_ic": _correlation(true, pred),
        "spearman_ic": _correlation(_average_rank(true), _average_rank(pred)),
        "mae": float(np.mean(np.abs(pred - true))),
        "rmse": float(np.sqrt(np.mean(np.square(pred - true)))),
        "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(true))),
    }


def time_thin_mask(
    timestamps: Sequence[float],
    day_ids: Sequence[Hashable],
    cooldown_seconds: float,
    *,
    candidate_mask: Sequence[bool] | None = None,
) -> np.ndarray:
    """Keep at most one candidate per cooldown interval, resetting each day."""

    if not np.isfinite(cooldown_seconds) or cooldown_seconds < 0:
        raise ValueError("cooldown_seconds must be finite and nonnegative")
    times = np.asarray(timestamps, dtype=float)
    days = np.asarray(day_ids)
    if times.ndim != 1 or days.ndim != 1 or times.shape != days.shape:
        raise ValueError("timestamps and day_ids must be aligned one-dimensional arrays")
    validate_day_sequence(days, times)
    candidates = np.ones(times.size, dtype=bool)
    if candidate_mask is not None:
        candidates = np.asarray(candidate_mask, dtype=bool)
        if candidates.shape != times.shape:
            raise ValueError("candidate_mask must align with timestamps")

    selected = np.zeros(times.size, dtype=bool)
    last_time: float | None = None
    last_day: Hashable | None = None
    for index in np.flatnonzero(candidates):
        day = days[index].item() if hasattr(days[index], "item") else days[index]
        if day != last_day:
            last_day = day
            last_time = None
        if last_time is None or times[index] - last_time >= cooldown_seconds:
            selected[index] = True
            last_time = float(times[index])
    return selected


def evaluate_with_thinning(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    timestamps: Sequence[float],
    day_ids: Sequence[Hashable],
    cooldown_seconds: float,
) -> dict[str, dict[str, float | int]]:
    """Report dense and cooldown-thinned metrics side by side."""

    true, pred, finite = _aligned_finite(y_true, y_pred)
    selected = time_thin_mask(
        timestamps, day_ids, cooldown_seconds, candidate_mask=finite
    )
    return {
        "dense": regression_metrics(true, pred),
        "thinned": regression_metrics(true[selected], pred[selected]),
    }


def validation_tail_threshold(
    validation_predictions: Sequence[float], quantile: float = 0.95
) -> float:
    """Calibrate a symmetric activation threshold using validation only."""

    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    predictions = np.asarray(validation_predictions, dtype=float)
    finite = np.abs(predictions[np.isfinite(predictions)])
    if finite.size == 0:
        raise ValueError("validation_predictions contain no finite values")
    return float(np.quantile(finite, quantile))


def cost_sensitivity(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    timestamps: Sequence[float],
    day_ids: Sequence[Hashable],
    *,
    activation_threshold: float,
    cooldown_seconds: float,
    round_trip_costs_bps: Iterable[float] = (0.0, 0.5, 1.0, 2.0),
) -> dict:
    """Evaluate side-adjusted returns under simple round-trip cost scenarios.

    This is a diagnostic, not an execution backtest.  The threshold should be
    obtained from a validation set with :func:`validation_tail_threshold`.
    """

    if not np.isfinite(activation_threshold) or activation_threshold < 0:
        raise ValueError("activation_threshold must be finite and nonnegative")
    true, pred, finite = _aligned_finite(y_true, y_pred)
    candidates = finite & (np.abs(pred) >= activation_threshold) & (pred != 0)
    selected = time_thin_mask(
        timestamps, day_ids, cooldown_seconds, candidate_mask=candidates
    )
    gross = np.sign(pred[selected]) * true[selected]

    scenarios: dict[str, dict[str, float]] = {}
    for raw_cost in round_trip_costs_bps:
        cost = float(raw_cost)
        if not np.isfinite(cost) or cost < 0:
            raise ValueError("round_trip_costs_bps must be finite and nonnegative")
        net = gross - cost * 1e-4
        scenarios[f"{cost:g}"] = {
            "mean_net_return": float(np.mean(net)) if net.size else float("nan"),
            "mean_net_bps": float(np.mean(net) * 1e4) if net.size else float("nan"),
            "net_hit_rate": float(np.mean(net > 0)) if net.size else float("nan"),
        }

    return {
        "selected_signals": int(gross.size),
        "activation_threshold": float(activation_threshold),
        "cooldown_seconds": float(cooldown_seconds),
        "gross_mean_return": float(np.mean(gross)) if gross.size else float("nan"),
        "gross_mean_bps": float(np.mean(gross) * 1e4) if gross.size else float("nan"),
        "gross_hit_rate": float(np.mean(gross > 0)) if gross.size else float("nan"),
        "cost_scenarios_bps": scenarios,
    }

