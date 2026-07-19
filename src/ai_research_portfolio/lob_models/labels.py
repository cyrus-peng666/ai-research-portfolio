"""Leakage-aware targets, normalization, windows, and chronological splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


def _as_1d(name: str, values: Sequence) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def validate_day_sequence(day_ids: Sequence[Hashable], timestamps: Sequence[float] | None = None) -> None:
    """Require contiguous day blocks and nondecreasing timestamps within a day."""

    days = _as_1d("day_ids", day_ids)
    if days.size == 0:
        raise ValueError("day_ids cannot be empty")

    seen: set[Hashable] = set()
    previous: Hashable | None = None
    for raw_day in days.tolist():
        day = raw_day.item() if hasattr(raw_day, "item") else raw_day
        if day != previous:
            if day in seen:
                raise ValueError("each day must appear in one contiguous block")
            seen.add(day)
            previous = day

    if timestamps is None:
        return
    times = _as_1d("timestamps", timestamps).astype(float, copy=False)
    if times.shape != days.shape or not np.isfinite(times).all():
        raise ValueError("timestamps must be finite and aligned with day_ids")
    for start, stop in _day_blocks(days):
        if np.any(np.diff(times[start:stop]) < 0):
            raise ValueError("timestamps must be nondecreasing within each day")


def _day_blocks(day_ids: np.ndarray):
    starts = np.r_[0, np.flatnonzero(day_ids[1:] != day_ids[:-1]) + 1]
    stops = np.r_[starts[1:], day_ids.size]
    return zip(starts.tolist(), stops.tolist())


def strict_future_average_return(
    mid_prices: Sequence[float],
    timestamps: Sequence[float],
    day_ids: Sequence[Hashable],
    horizon_seconds: float,
    *,
    log_return: bool = False,
) -> np.ndarray:
    """Compute a target containing only prices strictly after decision time.

    For event ``i``, the future reference is the mean of observations in
    ``(t_i, t_i + horizon_seconds]``.  A target is missing when the full
    horizon is unavailable, the future set is empty, or any required price is
    invalid.  A future window never crosses a day boundary.
    """

    if not np.isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive and finite")
    mid = _as_1d("mid_prices", mid_prices).astype(float, copy=False)
    times = _as_1d("timestamps", timestamps).astype(float, copy=False)
    days = _as_1d("day_ids", day_ids)
    if not (mid.shape == times.shape == days.shape):
        raise ValueError("mid_prices, timestamps, and day_ids must align")
    validate_day_sequence(days, times)

    target = np.full(mid.shape, np.nan, dtype=float)
    for start, stop in _day_blocks(days):
        day_times = times[start:stop]
        day_mid = mid[start:stop]
        finite_positive = np.isfinite(day_mid) & (day_mid > 0)
        sums = np.r_[0.0, np.cumsum(np.where(finite_positive, day_mid, 0.0))]
        counts = np.r_[0, np.cumsum(finite_positive.astype(np.int64))]

        for local_i, current_time in enumerate(day_times):
            horizon_end = current_time + horizon_seconds
            if day_times[-1] < horizon_end:
                continue
            # Several order-book events can share one exchange timestamp.  A
            # strict time-based target must exclude every observation at
            # ``current_time``, not only the current row.
            future_start = int(np.searchsorted(day_times, current_time, side="right"))
            future_stop = int(np.searchsorted(day_times, horizon_end, side="right"))
            required = future_stop - future_start
            if required <= 0 or not finite_positive[local_i]:
                continue
            valid = int(counts[future_stop] - counts[future_start])
            if valid != required:
                continue
            future_mean = (sums[future_stop] - sums[future_start]) / required
            current_mid = day_mid[local_i]
            if log_return:
                target[start + local_i] = np.log(future_mean / current_mid)
            else:
                target[start + local_i] = future_mean / current_mid - 1.0
    return target


@dataclass
class TrainFittedStandardizer:
    """Feature-wise standardizer that can only be fitted on an explicit mask."""

    epsilon: float = 1e-8
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray, train_mask: Sequence[bool]) -> "TrainFittedStandardizer":
        array = np.asarray(values, dtype=float)
        mask = _as_1d("train_mask", train_mask).astype(bool, copy=False)
        if array.ndim < 2 or array.shape[0] != mask.size:
            raise ValueError("values must be (N, ..., features) and align with train_mask")
        if not mask.any():
            raise ValueError("train_mask must select at least one row")
        flattened = array[mask].reshape(-1, array.shape[-1])
        if not np.isfinite(flattened).all():
            raise ValueError("training values must be finite")
        self.mean_ = flattened.mean(axis=0)
        self.scale_ = flattened.std(axis=0)
        self.scale_ = np.where(self.scale_ < self.epsilon, 1.0, self.scale_)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before transform")
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != self.mean_.size:
            raise ValueError("feature dimension differs from fitted data")
        shape = (1,) * (array.ndim - 1) + (self.mean_.size,)
        return (array - self.mean_.reshape(shape)) / self.scale_.reshape(shape)

    def fit_transform(self, values: np.ndarray, train_mask: Sequence[bool]) -> np.ndarray:
        return self.fit(values, train_mask).transform(values)


def past_only_standardize_by_day(
    values: np.ndarray,
    day_ids: Sequence[Hashable],
    *,
    lookback_days: int = 5,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Standardize each day using only preceding days.

    The first day is returned as ``NaN`` because no historical calibration is
    available.  This explicit missing state is safer than silently fitting on
    the day being transformed.
    """

    array = np.asarray(values, dtype=float)
    days = _as_1d("day_ids", day_ids)
    if array.ndim < 2 or array.shape[0] != days.size:
        raise ValueError("values must be (N, ..., features) and align with day_ids")
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    validate_day_sequence(days)

    output = np.full_like(array, np.nan, dtype=float)
    history: list[np.ndarray] = []
    for start, stop in _day_blocks(days):
        current = array[start:stop]
        if history:
            reference = np.concatenate(history[-lookback_days:], axis=0)
            flat = reference.reshape(-1, array.shape[-1])
            if not np.isfinite(flat).all():
                raise ValueError("historical normalization values must be finite")
            mean = flat.mean(axis=0)
            scale = flat.std(axis=0)
            scale = np.where(scale < epsilon, 1.0, scale)
            shape = (1,) * (array.ndim - 1) + (array.shape[-1],)
            output[start:stop] = (current - mean.reshape(shape)) / scale.reshape(shape)
        history.append(current)
    return output


@dataclass(frozen=True)
class DaySplit:
    train_mask: np.ndarray
    validation_mask: np.ndarray
    test_mask: np.ndarray
    train_days: tuple
    validation_days: tuple
    test_days: tuple


def chronological_day_split(
    day_ids: Sequence[Hashable],
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> DaySplit:
    """Create chronological masks without splitting a trading day."""

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave a positive test share")
    days = _as_1d("day_ids", day_ids)
    validate_day_sequence(days)
    unique_days = tuple(days[np.r_[True, days[1:] != days[:-1]]].tolist())
    if len(unique_days) < 3:
        raise ValueError("at least three complete days are required")

    n_days = len(unique_days)
    n_train = max(1, int(np.floor(n_days * train_fraction)))
    n_validation = max(1, int(np.floor(n_days * validation_fraction)))
    if n_train + n_validation >= n_days:
        n_train = n_days - n_validation - 1
    if n_train < 1:
        raise ValueError("fractions do not leave complete train/validation/test days")

    train_days = unique_days[:n_train]
    validation_days = unique_days[n_train : n_train + n_validation]
    test_days = unique_days[n_train + n_validation :]
    return DaySplit(
        train_mask=np.isin(days, train_days),
        validation_mask=np.isin(days, validation_days),
        test_mask=np.isin(days, test_days),
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
    )


def day_aware_window_end_indices(
    day_ids: Sequence[Hashable],
    window_size: int,
    *,
    valid_target: Sequence[bool] | None = None,
) -> np.ndarray:
    """Return endpoints whose complete input window stays within one day."""

    if window_size < 1:
        raise ValueError("window_size must be positive")
    days = _as_1d("day_ids", day_ids)
    validate_day_sequence(days)
    target_mask = np.ones(days.size, dtype=bool)
    if valid_target is not None:
        target_mask = _as_1d("valid_target", valid_target).astype(bool, copy=False)
        if target_mask.shape != days.shape:
            raise ValueError("valid_target must align with day_ids")

    endpoints: list[int] = []
    for start, stop in _day_blocks(days):
        first_end = start + window_size - 1
        if first_end >= stop:
            continue
        candidates = np.arange(first_end, stop)
        endpoints.extend(candidates[target_mask[candidates]].tolist())
    return np.asarray(endpoints, dtype=np.int64)


@dataclass(frozen=True)
class WindowedLOB:
    book: np.ndarray
    time_delta: np.ndarray
    target: np.ndarray
    end_indices: np.ndarray
    day_ids: np.ndarray


def build_day_aware_windows(
    book: np.ndarray,
    time_delta: Sequence[float],
    targets: Sequence[float],
    day_ids: Sequence[Hashable],
    window_size: int,
) -> WindowedLOB:
    """Materialize fixed windows while rejecting invalid or cross-day targets."""

    book_array = np.asarray(book)
    delta = _as_1d("time_delta", time_delta).astype(float, copy=False)
    target = _as_1d("targets", targets).astype(float, copy=False)
    days = _as_1d("day_ids", day_ids)
    if book_array.ndim != 4 or book_array.shape[1] != 2 or book_array.shape[-1] != 4:
        raise ValueError("book must have shape (N, 2, levels, 4)")
    if not (book_array.shape[0] == delta.size == target.size == days.size):
        raise ValueError("all inputs must share the same event count")
    if np.any(delta < 0) or not np.isfinite(delta).all():
        raise ValueError("time_delta must be finite and nonnegative")

    endpoints = day_aware_window_end_indices(
        days, window_size, valid_target=np.isfinite(target)
    )
    windows = np.asarray(
        [book_array[end - window_size + 1 : end + 1] for end in endpoints]
    )
    delta_windows = np.asarray(
        [delta[end - window_size + 1 : end + 1] for end in endpoints]
    )
    if endpoints.size == 0:
        windows = np.empty((0, window_size, *book_array.shape[1:]), dtype=book_array.dtype)
        delta_windows = np.empty((0, window_size), dtype=float)
    return WindowedLOB(
        book=windows,
        time_delta=delta_windows,
        target=target[endpoints],
        end_indices=endpoints,
        day_ids=days[endpoints],
    )
