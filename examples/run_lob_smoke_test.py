"""Run one leakage-aware DeepGELOB optimization step on synthetic data."""

from __future__ import annotations

import json

import numpy as np
import torch

from ai_research_portfolio.lob_models.deepgelob import DeepGELOB
from ai_research_portfolio.lob_models.evaluation import (
    cost_sensitivity,
    evaluate_with_thinning,
    validation_tail_threshold,
)
from ai_research_portfolio.lob_models.labels import (
    TrainFittedStandardizer,
    build_day_aware_windows,
    chronological_day_split,
    strict_future_average_return,
)
from ai_research_portfolio.lob_models.synthetic import generate_synthetic_lob


def main() -> None:
    torch.manual_seed(7)
    stream = generate_synthetic_lob(days=6, events_per_day=96, levels=10, seed=7)
    targets = strict_future_average_return(
        stream.mid_price,
        stream.timestamps,
        stream.day_ids,
        horizon_seconds=1.0,
    )
    event_split = chronological_day_split(stream.day_ids)
    normalized_book = TrainFittedStandardizer().fit_transform(
        stream.book, event_split.train_mask
    ).astype(np.float32)
    windows = build_day_aware_windows(
        normalized_book,
        np.log1p(stream.time_delta),
        targets,
        stream.day_ids,
        window_size=24,
    )
    window_split = chronological_day_split(windows.day_ids)

    train_index = np.flatnonzero(window_split.train_mask)[:32]
    book = torch.from_numpy(windows.book[train_index])
    delta = torch.from_numpy(windows.time_delta[train_index].astype(np.float32))
    target = torch.from_numpy(windows.target[train_index].astype(np.float32))

    model = DeepGELOB(levels=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    prediction = model(book, delta)
    loss = torch.nn.functional.huber_loss(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        all_predictions = model(
            torch.from_numpy(windows.book),
            torch.from_numpy(windows.time_delta.astype(np.float32)),
        ).numpy()
    endpoint_times = stream.timestamps[windows.end_indices]
    validation_predictions = all_predictions[window_split.validation_mask]
    threshold = validation_tail_threshold(validation_predictions, quantile=0.8)
    test = window_split.test_mask
    metrics = evaluate_with_thinning(
        windows.target[test],
        all_predictions[test],
        endpoint_times[test],
        windows.day_ids[test],
        cooldown_seconds=1.0,
    )
    costs = cost_sensitivity(
        windows.target[test],
        all_predictions[test],
        endpoint_times[test],
        windows.day_ids[test],
        activation_threshold=threshold,
        cooldown_seconds=1.0,
        round_trip_costs_bps=(0.0, 1.0, 2.0),
    )

    print(f"windows={len(windows.target)}, train_step_loss={loss.item():.6g}")
    print(json.dumps({"metrics": metrics, "costs": costs}, indent=2, allow_nan=True))
    print("Synthetic smoke test only; these numbers are not empirical results.")


if __name__ == "__main__":
    main()

