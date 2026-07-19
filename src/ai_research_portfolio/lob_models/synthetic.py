"""Small deterministic synthetic LOB generator for public smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticLOB:
    book: np.ndarray
    time_delta: np.ndarray
    timestamps: np.ndarray
    day_ids: np.ndarray
    mid_price: np.ndarray


def generate_synthetic_lob(
    *,
    days: int = 6,
    events_per_day: int = 160,
    levels: int = 10,
    seed: int = 7,
) -> SyntheticLOB:
    """Generate a toy event stream without using or mimicking licensed data."""

    if days < 1 or events_per_day < 4 or levels < 1:
        raise ValueError("days, events_per_day, and levels must be positive")
    rng = np.random.default_rng(seed)
    total = days * events_per_day
    book = np.empty((total, 2, levels, 4), dtype=np.float32)
    time_delta = np.empty(total, dtype=np.float32)
    timestamps = np.empty(total, dtype=np.float64)
    day_ids = np.repeat(np.arange(days), events_per_day)
    mid_price = np.empty(total, dtype=np.float64)

    cursor = 0
    reference_price = 100.0
    level_index = np.arange(1, levels + 1, dtype=float)
    for day in range(days):
        delta = rng.exponential(scale=0.35, size=events_per_day)
        delta[0] = 0.0
        times = 34_200.0 + np.cumsum(delta)
        latent = np.empty(events_per_day, dtype=float)
        latent[0] = rng.normal(scale=0.5)
        for index in range(1, events_per_day):
            latent[index] = 0.92 * latent[index - 1] + rng.normal(scale=0.28)

        innovations = 1e-4 * (0.10 * latent + rng.normal(scale=0.7, size=events_per_day))
        mids = reference_price * np.exp(np.cumsum(innovations))
        reference_price = float(mids[-1] * np.exp(rng.normal(scale=2e-4)))

        for local in range(events_per_day):
            global_index = cursor + local
            spread = 0.01 + 0.002 * (1.0 + np.tanh(abs(latent[local])))
            tick = 0.01
            bid_prices = mids[local] - spread / 2.0 - tick * (level_index - 1)
            ask_prices = mids[local] + spread / 2.0 + tick * (level_index - 1)
            imbalance = np.tanh(latent[local])
            depth_curve = 250.0 + 45.0 * level_index
            bid_volume = depth_curve * np.exp(0.28 * imbalance + rng.normal(0, 0.08, levels))
            ask_volume = depth_curve * np.exp(-0.28 * imbalance + rng.normal(0, 0.08, levels))
            bid_count = np.maximum(1.0, rng.poisson(np.maximum(bid_volume / 70.0, 1.0)))
            ask_count = np.maximum(1.0, rng.poisson(np.maximum(ask_volume / 70.0, 1.0)))
            bid_age = rng.exponential(1.2 + 0.25 * (1.0 + imbalance), levels)
            ask_age = rng.exponential(1.2 + 0.25 * (1.0 - imbalance), levels)

            # side 0 = bid, side 1 = ask; channel = P, V, N, A.
            book[global_index, 0, :, :] = np.stack(
                (bid_prices, bid_volume, bid_count, bid_age), axis=-1
            )
            book[global_index, 1, :, :] = np.stack(
                (ask_prices, ask_volume, ask_count, ask_age), axis=-1
            )

        sl = slice(cursor, cursor + events_per_day)
        time_delta[sl] = delta
        timestamps[sl] = times
        mid_price[sl] = mids
        cursor += events_per_day

    return SyntheticLOB(
        book=book,
        time_delta=time_delta,
        timestamps=timestamps,
        day_ids=day_ids,
        mid_price=mid_price,
    )

