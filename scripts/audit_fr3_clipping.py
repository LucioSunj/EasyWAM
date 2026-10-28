#!/usr/bin/env python3
"""Read-only audit of FR3 action/state clipping and trajectory continuity.

The audit mirrors the active EasyWAM contract without importing the trainer:
raw actions are absolute targets, the first seven action dimensions are made
relative to the window's first state, and global z-score normalization is
clamped to ``[-5, 5]``.  It emits JSON/CSV evidence only; it never modifies a
dataset, checkpoint, stats file, or training configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised only on an incomplete host
    pq = None


DIM_NAMES = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays into stable JSON-native values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _as_float_array(value: Any, *, name: str, ndim: int = 1) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.shape}")
    return array


def load_global_stats(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Load global mean/std while preserving the source JSON for provenance."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    stats: dict[str, dict[str, np.ndarray]] = {}
    for field in ("state", "action"):
        try:
            source = raw[field]["default"]
            mean = _as_float_array(source["global_mean"], name=f"{field}.global_mean")
            std = _as_float_array(source["global_std"], name=f"{field}.global_std")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"stats file lacks global {field} mean/std") from exc
        if mean.shape != std.shape or mean.shape != (8,):
            raise ValueError(f"{field} stats must be eight-dimensional, got {mean.shape}/{std.shape}")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
            raise ValueError(f"{field} stats contain non-finite or negative values")
        stats[field] = {"mean": mean, "std": std}
    return raw, stats


def zscore(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Match SingleFieldLinearNormalizer's float64 algebra, including epsilon."""

    return values / (std + 1e-8) + (-mean / (std + 1e-8))


def normalize_and_error(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return z, clipped mask, and absolute raw-space clamp error."""

    z = zscore(values, mean, std)
    if threshold is None:
        return z, np.zeros_like(z, dtype=bool), np.zeros_like(values, dtype=np.float64)
    clipped_z = np.clip(z, -threshold, threshold)
    error = np.abs((clipped_z - z) * (std + 1e-8))
    return z, np.abs(z) > threshold, error


def _read_episode_tables(dataset: Path) -> list[dict[str, Any]]:
    if pq is None:
        raise RuntimeError("pyarrow is required to read the LeRobot parquet files")
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    metadata = {
        int(item["episode_index"]): int(item["length"])
        for line in (dataset / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in [json.loads(line)]
    }
    if int(info["total_episodes"]) != len(metadata):
        raise ValueError("info.json and episodes.jsonl disagree on episode count")

    columns = ["observation.state", "action", "episode_index", "frame_index", "timestamp"]
    episodes: list[dict[str, Any]] = []
    paths = sorted((dataset / "data").rglob("*.parquet"))
    if len(paths) != len(metadata):
        raise ValueError(f"expected {len(metadata)} parquet files, found {len(paths)}")
    for path in paths:
        table = pq.read_table(path, columns=columns)
        episode_ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        if len(np.unique(episode_ids)) != 1:
            raise ValueError(f"multiple episode ids in {path}")
        episode_id = int(episode_ids[0])
        order = np.argsort(np.asarray(table["frame_index"].to_numpy(), dtype=np.int64))
        frame = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)[order]
        timestamp = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)[order]
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)[order]
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)[order]
        if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 8:
            raise ValueError(f"unexpected state/action shape in {path}: {state.shape}/{action.shape}")
        expected_length = metadata[episode_id]
        if len(frame) != expected_length or not np.array_equal(frame, np.arange(expected_length)):
            raise ValueError(f"non-contiguous frame metadata in {path}")
        if not np.allclose(timestamp, frame / float(info["fps"]), atol=1e-4, rtol=0):
            raise ValueError(f"timestamps do not match frame/fps in {path}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"non-finite state/action values in {path}")
        episodes.append(
            {
                "episode_index": episode_id,
                "frame": frame,
                "timestamp": timestamp,
                "state": state,
                "action": action,
                "path": str(path),
            }
        )
    episodes.sort(key=lambda item: item["episode_index"])
    if [item["episode_index"] for item in episodes] != sorted(metadata):
        raise ValueError("parquet episode coverage is incomplete")
    return episodes


def _continuity_reference(episodes: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Measure neighbor continuity; thresholds are empirical p99/p999 only."""

    result: dict[str, dict[str, Any]] = {}
    for field in ("state", "action"):
        values = [np.diff(item[field][:, :7], axis=0) for item in episodes if len(item[field]) > 1]
        deltas = np.concatenate(values, axis=0) if values else np.empty((0, 7), dtype=np.float64)
        max_abs = np.max(np.abs(deltas), axis=1) if len(deltas) else np.empty(0, dtype=np.float64)
        result[field] = {
            "n_neighbor_steps": int(len(max_abs)),
            "max_abs_joint_delta_p50": float(np.quantile(max_abs, 0.50)) if len(max_abs) else None,
            "max_abs_joint_delta_p99": float(np.quantile(max_abs, 0.99)) if len(max_abs) else None,
            "max_abs_joint_delta_p999": float(np.quantile(max_abs, 0.999)) if len(max_abs) else None,
            "max_abs_joint_delta_max": float(max_abs.max()) if len(max_abs) else None,
        }
    return result


def _continuity_at(
    values: np.ndarray,
    frame: int,
    reference: dict[str, Any],
) -> dict[str, Any]:
    if len(values) <= 1:
        return {"prev_max_abs": None, "next_max_abs": None, "max_abs": None, "class": "no_neighbor"}
    deltas = np.diff(values[:, :7], axis=0)
    prev = float(np.max(np.abs(deltas[frame - 1]))) if frame > 0 else None
    nxt = float(np.max(np.abs(deltas[frame]))) if frame + 1 < len(values) else None
    candidates = [value for value in (prev, nxt) if value is not None]
    observed = max(candidates) if candidates else None
    p99 = reference.get("max_abs_joint_delta_p99")
    p999 = reference.get("max_abs_joint_delta_p999")
    if observed is None or p99 is None or p999 is None:
        label = "no_reference"
    elif observed <= p99:
        label = "within_p99"
    elif observed <= p999:
        label = "between_p99_p999"
    else:
        label = "beyond_p999"
    return {
        "prev_max_abs": prev,
        "next_max_abs": nxt,
        "max_abs": observed,
        "p99_ratio": (observed / p99) if observed is not None and p99 and p99 > 0 else None,
        "class": label,
    }


def _contiguous_runs(frames: Iterable[int]) -> list[dict[str, int]]:
    values = sorted(set(int(frame) for frame in frames))
    if not values:
        return []
    runs: list[dict[str, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            runs.append({"start_frame": start, "end_frame": previous, "length": previous - start + 1})
            start = value
        previous = value
    runs.append({"start_frame": start, "end_frame": previous, "length": previous - start + 1})
    return runs


def _fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.10g}"


def _video_paths(dataset: Path, episode: int) -> dict[str, str]:
    return {
        camera: str(dataset / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{episode:06d}.mp4")
        for camera in ("left", "wrist")
    }


def _evaluate_matrix(
    matrix: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float | None,
) -> dict[str, Any]:
    z, clipped, error = normalize_and_error(matrix, mean, std, threshold)
    dim_counts = clipped.sum(axis=0).astype(np.int64)
    vector_mask = clipped.any(axis=1)
    return {
        "vectors": int(len(matrix)),
        "clipped_vectors": int(vector_mask.sum()),
        "clipped_dimensions": dim_counts.tolist(),
        "clip_fraction_dimensions": (dim_counts / max(len(matrix), 1)).tolist(),
        "max_abs_z_per_dim": np.max(np.abs(z), axis=0).tolist() if len(matrix) else [None] * matrix.shape[1],
        "roundtrip_abs_error_sum_per_dim": error.sum(axis=0).tolist(),
        "roundtrip_abs_error_max_per_dim": np.max(error, axis=0).tolist() if len(matrix) else [None] * matrix.shape[1],
        "roundtrip_vectors_error_gt_1e-5": int((error > 1e-5).any(axis=1).sum()),
        "roundtrip_abs_error_sum": float(error.sum()),
        "roundtrip_abs_error_max": float(error.max()) if error.size else 0.0,
        "threshold": threshold,
    }


def audit_episodes(
    episodes: Sequence[dict[str, Any]],
    stats: dict[str, dict[str, np.ndarray]],
    dataset_root: Path,
    *,
    horizon: int = 16,
    horizons: Sequence[int] = (1, 4, 8, 16, 32),
    clip_threshold: float = 5.0,
    joint_dim: int = 7,
    top_k: int = 100,
) -> dict[str, Any]:
    if horizon <= 0 or joint_dim != 7:
        raise ValueError("horizon must be positive and this audit expects joint_dim=7")
    if clip_threshold <= 0:
        raise ValueError("clip threshold must be positive")
    horizon_values = sorted(set(int(item) for item in horizons) | {int(horizon)})

    continuity = _continuity_reference(episodes)
    episode_by_id = {int(item["episode_index"]): item for item in episodes}
    state_matrix = np.concatenate([item["state"] for item in episodes], axis=0)
    state_z, state_clipped, state_error = normalize_and_error(
        state_matrix, stats["state"]["mean"], stats["state"]["std"], clip_threshold
    )
    state_clip_by_dim = state_clipped.sum(axis=0).astype(np.int64)
    state_clip_events: list[dict[str, Any]] = []
    state_cursor = 0
    for episode in episodes:
        ep = int(episode["episode_index"])
        state_ep_z = state_z[state_cursor : state_cursor + len(episode["state"])]
        state_ep_clipped = state_clipped[state_cursor : state_cursor + len(episode["state"])]
        for frame in np.flatnonzero(state_ep_clipped.any(axis=1)):
            frame = int(frame)
            dims = np.flatnonzero(state_ep_clipped[frame]).astype(int).tolist()
            state_clip_events.append(
                {
                    "episode_index": ep,
                    "frame": frame,
                    "timestamp": float(episode["timestamp"][frame]),
                    "clipped_dimensions": dims,
                    "clipped_dimension_names": [DIM_NAMES[index] for index in dims],
                    "raw_state": episode["state"][frame].tolist(),
                    "z": state_ep_z[frame].tolist(),
                    "roundtrip_abs_error": state_error[state_cursor + frame].tolist(),
                }
            )
        state_cursor += len(episode["state"])

    action_mean = stats["action"]["mean"]
    action_std = stats["action"]["std"]
    all_relative: list[np.ndarray] = []
    all_relative_padded: list[np.ndarray] = []
    episode_relative_stats: list[dict[str, Any]] = []
    active_action_episode_means: list[np.ndarray] = []
    active_action_episode_vars: list[np.ndarray] = []
    active_state_episode_means: list[np.ndarray] = []
    active_state_episode_vars: list[np.ndarray] = []
    horizon_results: dict[str, dict[str, Any]] = {}
    clip_events: list[dict[str, Any]] = []
    event_frame_dims: dict[tuple[int, int], set[int]] = defaultdict(set)
    event_frame_starts: dict[tuple[int, int], set[int]] = defaultdict(set)
    clip_windows: list[dict[str, Any]] = []
    clip_counts_by_episode: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(8, dtype=np.int64))
    continuity_classes: Counter[str] = Counter()

    for current_horizon in horizon_values:
        result = {
            "horizon": current_horizon,
            "windows": 0,
            "valid_vectors": 0,
            "clipped_windows": 0,
            "clipped_vectors": 0,
            "clipped_dimensions": np.zeros(8, dtype=np.int64),
            "unique_target_frames": 0,
            "unique_target_frames_by_dim": np.zeros(8, dtype=np.int64),
            "episodes_with_clips": set(),
            "roundtrip_abs_error_sum_per_dim": np.zeros(8, dtype=np.float64),
            "roundtrip_abs_error_max_per_dim": np.zeros(8, dtype=np.float64),
            "roundtrip_vectors_error_gt_1e-5": 0,
        }
        target_frames: set[tuple[int, int]] = set()
        target_frames_by_dim: list[set[tuple[int, int]]] = [set() for _ in range(8)]
        for episode in episodes:
            ep = int(episode["episode_index"])
            state = episode["state"]
            action = episode["action"]
            length = len(state)
            episode_rel: list[np.ndarray] = []
            for start in range(length):
                target = np.arange(start, min(length, start + current_horizon), dtype=np.int64)
                relative = action[target].copy()
                relative[:, :joint_dim] -= state[start, :joint_dim]
                if current_horizon == horizon:
                    episode_rel.append(relative)
                z, clipped, error = normalize_and_error(relative, action_mean, action_std, clip_threshold)
                result["windows"] += 1
                result["valid_vectors"] += len(target)
                result["clipped_dimensions"] += clipped.sum(axis=0).astype(np.int64)
                result["roundtrip_abs_error_sum_per_dim"] += error.sum(axis=0)
                result["roundtrip_abs_error_max_per_dim"] = np.maximum(
                    result["roundtrip_abs_error_max_per_dim"], error.max(axis=0)
                )
                result["roundtrip_vectors_error_gt_1e-5"] += int((error > 1e-5).any(axis=1).sum())
                clipped_vector_mask = clipped.any(axis=1)
                if clipped_vector_mask.any():
                    result["clipped_windows"] += 1
                    result["clipped_vectors"] += int(clipped_vector_mask.sum())
                    result["episodes_with_clips"].add(ep)
                for row, frame in enumerate(target):
                    dims = np.flatnonzero(clipped[row]).astype(int).tolist()
                    if not dims:
                        continue
                    key = (ep, int(frame))
                    target_frames.add(key)
                    for dim in dims:
                        target_frames_by_dim[dim].add(key)
                    if current_horizon != horizon:
                        continue
                    for dim in dims:
                        event_frame_dims[key].add(dim)
                        event_frame_starts[key].add(start)
                    action_same_frame = action[frame] - state[frame]
                    anchor_delta = state[frame] - state[start]
                    state_cont = _continuity_at(state, int(frame), continuity["state"])
                    action_cont = _continuity_at(action, int(frame), continuity["action"])
                    label = f"state:{state_cont['class']}|action:{action_cont['class']}"
                    continuity_classes[label] += 1
                    clip_events.append(
                        {
                            "episode_index": ep,
                            "start_frame": int(start),
                            "target_frame": int(frame),
                            "target_timestamp": float(episode["timestamp"][frame]),
                            "start_timestamp": float(episode["timestamp"][start]),
                            "horizon": int(horizon),
                            "clipped_dimensions": dims,
                            "clipped_dimension_names": [DIM_NAMES[index] for index in dims],
                            "raw_target": action[frame].tolist(),
                            "anchor_state": state[start].tolist(),
                            "target_state": state[frame].tolist(),
                            "relative_target": relative[row].tolist(),
                            "z": z[row].tolist(),
                            "roundtrip_abs_error": error[row].tolist(),
                            "same_frame_action_minus_state": action_same_frame.tolist(),
                            "anchor_to_target_state_delta": anchor_delta.tolist(),
                            "state_neighbor_continuity": state_cont,
                            "action_neighbor_continuity": action_cont,
                            "video_paths": _video_paths(dataset_root, ep),
                            "source_parquet": episode["path"],
                        }
                    )
                    clip_counts_by_episode[ep] += clipped[row].astype(np.int64)

                if current_horizon == horizon and clipped_vector_mask.any():
                    clipped_positions = np.flatnonzero(clipped_vector_mask)
                    worst_row = int(clipped_positions[np.argmax(np.max(np.abs(z[clipped_positions]), axis=1))])
                    clip_windows.append(
                        {
                            "episode_index": ep,
                            "start_frame": int(start),
                            "end_frame_exclusive": int(min(length, start + current_horizon)),
                            "start_timestamp": float(episode["timestamp"][start]),
                            "end_timestamp_inclusive": float(episode["timestamp"][target[-1]]),
                            "horizon": int(horizon),
                            "clipped_vectors": int(clipped_vector_mask.sum()),
                            "clipped_dimensions": int(clipped.sum()),
                            "clipped_dimension_counts": clipped.sum(axis=0).astype(int).tolist(),
                            "max_abs_z": float(np.max(np.abs(z[clipped_vector_mask]))),
                            "max_roundtrip_abs_error": float(np.max(error[clipped_vector_mask])),
                            "worst_target_frame": int(target[worst_row]),
                            "worst_target_z": z[worst_row].tolist(),
                            "video_paths": _video_paths(dataset_root, ep),
                            "source_parquet": episode["path"],
                        }
                    )
            if current_horizon == horizon:
                episode_matrix = np.concatenate(episode_rel, axis=0) if episode_rel else np.empty((0, 8))
                all_relative.append(episode_matrix)
                ep_mean = episode_matrix.mean(axis=0) if len(episode_matrix) else np.zeros(8)
                ep_std = episode_matrix.std(axis=0) if len(episode_matrix) else np.zeros(8)
                # Dataset stats are computed by get_dataset_stats() over a
                # [episode, start, horizon] grid, repeating the terminal
                # target at each episode boundary.  Reconstruct that grid
                # separately from the valid (padding-excluded) audit above.
                length = len(state)
                padded_target = np.minimum(
                    np.arange(length, dtype=np.int64)[:, None] + np.arange(horizon, dtype=np.int64)[None, :],
                    length - 1,
                )
                padded_relative = action[padded_target].copy()
                padded_relative[:, :, :joint_dim] -= state[:, None, :joint_dim]
                padded_flat = padded_relative.reshape(-1, 8)
                all_relative_padded.append(padded_flat)
                # The source implementation calls cur_action.mean(0) and
                # cur_action.var(0), retaining the horizon axis, then pools
                # those per-step statistics equally across episodes/steps.
                active_action_episode_means.append(padded_relative.mean(axis=0))
                active_action_episode_vars.append(
                    padded_relative.var(axis=0, ddof=1) if len(padded_relative) > 1 else np.zeros((horizon, 8))
                )
                active_state_episode_means.append(state.mean(axis=0))
                active_state_episode_vars.append(state.var(axis=0, ddof=1) if len(state) > 1 else np.zeros(8))
                episode_relative_stats.append(
                    {
                        "episode_index": ep,
                        "vectors": int(len(episode_matrix)),
                        "mean": ep_mean.tolist(),
                        "std": ep_std.tolist(),
                    }
                )

        result["clipped_dimensions"] = result["clipped_dimensions"].astype(int).tolist()
        result["clip_fraction_dimensions"] = (
            np.asarray(result["clipped_dimensions"], dtype=np.float64) / max(result["valid_vectors"], 1)
        ).tolist()
        result["clipped_vector_fraction"] = result["clipped_vectors"] / max(result["valid_vectors"], 1)
        result["clipped_window_fraction"] = result["clipped_windows"] / max(result["windows"], 1)
        result["unique_target_frames"] = len(target_frames)
        result["unique_target_frames_by_dim"] = [len(values) for values in target_frames_by_dim]
        result["episodes_with_clips"] = sorted(result["episodes_with_clips"])
        result["episodes_with_clips_count"] = len(result["episodes_with_clips"])
        result["roundtrip_abs_error_sum_per_dim"] = result["roundtrip_abs_error_sum_per_dim"].tolist()
        result["roundtrip_abs_error_max_per_dim"] = result["roundtrip_abs_error_max_per_dim"].tolist()
        result["roundtrip_abs_error_sum"] = float(sum(result["roundtrip_abs_error_sum_per_dim"]))
        result["roundtrip_abs_error_max"] = float(max(result["roundtrip_abs_error_max_per_dim"], default=0.0))
        horizon_results[str(current_horizon)] = result

    all_relative_matrix = np.concatenate(all_relative, axis=0)
    all_relative_padded_matrix = np.concatenate(all_relative_padded, axis=0)
    pooled_mean = all_relative_matrix.mean(axis=0)
    pooled_std = all_relative_matrix.std(axis=0)
    episode_avg_mean = np.mean(np.asarray([item["mean"] for item in episode_relative_stats]), axis=0)
    episode_avg_std = np.mean(np.asarray([item["std"] for item in episode_relative_stats]), axis=0)
    active_action_step_means = np.asarray(active_action_episode_means)
    active_action_step_vars = np.asarray(active_action_episode_vars)
    active_action_mean_reconstructed = np.mean(active_action_step_means, axis=(0, 1))
    active_action_std_reconstructed = np.sqrt(
        np.mean(
            active_action_step_vars
            + (active_action_step_means - active_action_mean_reconstructed) ** 2,
            axis=(0, 1),
        )
    )
    active_state_mean_reconstructed = np.mean(np.asarray(active_state_episode_means), axis=0)
    active_state_std_reconstructed = np.sqrt(
        np.mean(
            np.asarray(active_state_episode_vars)
            + (np.asarray(active_state_episode_means) - active_state_mean_reconstructed) ** 2,
            axis=0,
        )
    )
    padded_pooled_mean = all_relative_padded_matrix.mean(axis=0)
    padded_pooled_std = all_relative_padded_matrix.std(axis=0)
    alternative_normalization = {
        "active_global_zscore_clamp5": _evaluate_matrix(all_relative_matrix, action_mean, action_std, 5.0),
        "active_global_zscore_unclipped": _evaluate_matrix(all_relative_matrix, action_mean, action_std, None),
        "active_global_zscore_clamp3": _evaluate_matrix(all_relative_matrix, action_mean, action_std, 3.0),
        "active_global_zscore_clamp8": _evaluate_matrix(all_relative_matrix, action_mean, action_std, 8.0),
        "recomputed_pooled_zscore_clamp5": _evaluate_matrix(all_relative_matrix, pooled_mean, pooled_std, 5.0),
        "unweighted_episode_average_mean_std_clamp5": _evaluate_matrix(
            all_relative_matrix, episode_avg_mean, episode_avg_std, 5.0
        ),
    }

    clip_events.sort(key=lambda item: (-max(abs(float(value)) for value in item["z"]), item["episode_index"], item["target_frame"], item["start_frame"]))
    clip_windows.sort(
        key=lambda item: (-item["clipped_dimensions"], -item["max_abs_z"], -item["max_roundtrip_abs_error"], item["episode_index"], item["start_frame"])
    )

    runs_by_episode_dim: dict[str, list[dict[str, int]]] = {}
    for (ep, frame), dims in sorted(event_frame_dims.items()):
        for dim in dims:
            runs_by_episode_dim.setdefault(f"{ep}:{dim}", []).append(frame)
    clip_runs = []
    for key, frames in sorted(runs_by_episode_dim.items(), key=lambda item: tuple(int(part) for part in item[0].split(":"))):
        ep, dim = (int(part) for part in key.split(":"))
        for run in _contiguous_runs(frames):
            run.update(
                {
                    "episode_index": ep,
                    "dimension": dim,
                    "dimension_name": DIM_NAMES[dim],
                    "start_timestamp": float(episode_by_id[ep]["timestamp"][run["start_frame"]]),
                    "end_timestamp": float(episode_by_id[ep]["timestamp"][run["end_frame"]]),
                    "window_starts_covering_run": sorted(
                        {start for frame in range(run["start_frame"], run["end_frame"] + 1) for start in event_frame_starts[(ep, frame)]}
                    ),
                }
            )
            clip_runs.append(run)

    gripper_state = state_matrix[:, 7]
    gripper_values, gripper_counts = np.unique(gripper_state, return_counts=True)
    gripper_action = np.concatenate([item["action"][:, 7] for item in episodes], axis=0)
    state_gripper_z = state_z[:, 7]
    state_gripper_clipped = state_clipped[:, 7]
    state_gripper_codes = {}
    for raw in gripper_values:
        mask = np.isclose(gripper_state, raw, atol=1e-7, rtol=0)
        state_gripper_codes[_fmt_float(float(raw))] = {
            "count": int(mask.sum()),
            "forward_z_before_clamp": float(np.mean(state_gripper_z[mask])),
            "forward_z_after_clamp": float(np.mean(np.clip(state_gripper_z[mask], -clip_threshold, clip_threshold))),
            "backward_value_after_clamp": float(np.mean(stats["state"]["mean"][7] + np.clip(state_gripper_z[mask], -clip_threshold, clip_threshold) * (stats["state"]["std"][7] + 1e-8))),
            "clipped": bool(state_gripper_clipped[mask].all()),
        }
    gripper_mismatch = np.abs(gripper_action - gripper_state) > 1e-6
    gripper_transition_state = np.abs(np.concatenate([np.diff(item["state"][:, 7]) for item in episodes if len(item["state"]) > 1])) > 1e-6
    gripper_transition_action = np.abs(np.concatenate([np.diff(item["action"][:, 7]) for item in episodes if len(item["action"]) > 1])) > 1e-6

    gripper_mismatch_examples = []
    for episode in episodes:
        mismatch = np.abs(episode["action"][:, 7] - episode["state"][:, 7]) > 1e-6
        for frame in np.flatnonzero(mismatch)[:20]:
            frame = int(frame)
            gripper_mismatch_examples.append(
                {
                    "episode_index": int(episode["episode_index"]),
                    "frame": frame,
                    "timestamp": float(episode["timestamp"][frame]),
                    "state_gripper": float(episode["state"][frame, 7]),
                    "action_gripper": float(episode["action"][frame, 7]),
                }
            )
    recommendation = {
        "training_stats_or_source_was_not_modified": True,
        "physical_rollout_metrics_available": False,
        "clip_observation": "Action clipping is a mathematical forward-normalizer effect; neighbor continuity labels are empirical p99/p999 comparisons, not proof of a recording error or physical impossibility.",
        "retraining_decision_boundary": "Do not invalidate or retrain solely from these offline clips. Escalate only if video review confirms recording corruption or a separate rollout/evaluation test shows behavior loss; no such physical evidence is included here.",
        "gripper_observation": "State gripper saturation is recorded separately because raw state values are inspected for binary/continuous semantics; distinct post-clamp codes are not by themselves evidence that retraining is required.",
    }

    result = {
        "audit_version": 1,
        "dataset_root": str(dataset_root),
        "dataset_info": {
            "episodes": len(episodes),
            "frames": int(sum(len(item["state"]) for item in episodes)),
            "fps": 10,
            "horizon": int(horizon),
            "joint_dim": int(joint_dim),
            "padding_excluded": True,
            "windows_include_all_starts": True,
        },
        "normalization_contract": {
            "mode": "global_zscore",
            "clip_threshold": float(clip_threshold),
            "state_mean": stats["state"]["mean"].tolist(),
            "state_std": stats["state"]["std"].tolist(),
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
            "relative_action": "absolute action target minus window-start state for dimensions 0..6; gripper unchanged",
        },
        "continuity_reference": continuity,
        "state_clipping": {
            "frames_with_any_clip": int(state_clipped.any(axis=1).sum()),
            "clip_count_per_dim": state_clip_by_dim.tolist(),
            "clip_fraction_per_dim": (state_clip_by_dim / max(len(state_matrix), 1)).tolist(),
            "max_abs_z_per_dim": np.max(np.abs(state_z), axis=0).tolist(),
            "roundtrip_abs_error_sum_per_dim": state_error.sum(axis=0).tolist(),
            "roundtrip_abs_error_max_per_dim": state_error.max(axis=0).tolist(),
            "clip_events": state_clip_events,
        },
        "action_horizon_dependence": horizon_results,
        "baseline_clip_events": clip_events,
        "baseline_clip_windows_top": clip_windows[: max(0, int(top_k))],
        "baseline_clip_runs": clip_runs,
        "baseline_continuity_class_counts": dict(sorted(continuity_classes.items())),
        "per_episode_clip_counts": {
            str(ep): counts.astype(int).tolist() for ep, counts in sorted(clip_counts_by_episode.items())
        },
        "relative_action_stats": {
            "vectors": int(len(all_relative_matrix)),
            "padded_stats_vectors": int(len(all_relative_padded_matrix)),
            "recomputed_pooled_mean": pooled_mean.tolist(),
            "recomputed_pooled_std": pooled_std.tolist(),
            "padded_pooled_mean": padded_pooled_mean.tolist(),
            "padded_pooled_std": padded_pooled_std.tolist(),
            "active_mean_minus_recomputed_pooled": (action_mean - pooled_mean).tolist(),
            "active_std_minus_recomputed_pooled": (action_std - pooled_std).tolist(),
            "active_stats_contract": "equal episode-weighted mean/variance over each episode's [start, horizon] transformed action grid, with terminal target replication; source uses torch.var (unbiased per episode)",
            "active_action_mean_reconstructed": active_action_mean_reconstructed.tolist(),
            "active_action_std_reconstructed": active_action_std_reconstructed.tolist(),
            "active_action_mean_minus_reconstructed": (action_mean - active_action_mean_reconstructed).tolist(),
            "active_action_std_minus_reconstructed": (action_std - active_action_std_reconstructed).tolist(),
            "active_state_mean_reconstructed": active_state_mean_reconstructed.tolist(),
            "active_state_std_reconstructed": active_state_std_reconstructed.tolist(),
            "active_state_mean_minus_reconstructed": (stats["state"]["mean"] - active_state_mean_reconstructed).tolist(),
            "active_state_std_minus_reconstructed": (stats["state"]["std"] - active_state_std_reconstructed).tolist(),
            "unweighted_episode_average_mean": episode_avg_mean.tolist(),
            "unweighted_episode_average_std": episode_avg_std.tolist(),
            "episode_stats": episode_relative_stats,
        },
        "normalization_alternatives": alternative_normalization,
        "gripper_analysis": {
            "state_raw_min": float(gripper_state.min()),
            "state_raw_max": float(gripper_state.max()),
            "state_unique_values": [{"value": float(value), "count": int(count)} for value, count in zip(gripper_values, gripper_counts)],
            "state_is_binary_0_1_within_1e-6": bool(np.all(np.isclose(gripper_state, 0.0, atol=1e-6) | np.isclose(gripper_state, 1.0, atol=1e-6))),
            "state_transition_count": int(gripper_transition_state.sum()),
            "action_raw_min": float(gripper_action.min()),
            "action_raw_max": float(gripper_action.max()),
            "action_unique_values": [{"value": float(value), "count": int(count)} for value, count in zip(*np.unique(gripper_action, return_counts=True))],
            "action_is_binary_0_1_within_1e-6": bool(np.all(np.isclose(gripper_action, 0.0, atol=1e-6) | np.isclose(gripper_action, 1.0, atol=1e-6))),
            "action_transition_count": int(gripper_transition_action.sum()),
            "same_frame_action_state_mismatch_count": int(gripper_mismatch.sum()),
            "same_frame_action_state_mismatch_fraction": float(gripper_mismatch.mean()),
            "mismatch_examples_first_20_per_episode": gripper_mismatch_examples,
            "state_clip_count": int(state_gripper_clipped.sum()),
            "state_clip_episode_frame_count": int(sum(1 for item in state_clip_events if 7 in item["clipped_dimensions"])),
            "state_normalized_codes": state_gripper_codes,
            "action_clip_count": int((np.abs(zscore(np.asarray([item for item in gripper_action[:, None]]), np.array([action_mean[7]]), np.array([action_std[7]]))) > clip_threshold).sum()),
            "semantic_note": "Separate state/action counts preserve the distinction between binary gripper state and command; no continuous physical range is inferred from a binary sample.",
        },
        "recommendation": recommendation,
        "provenance": {
            "episode_source_paths": [item["path"] for item in episodes],
            "stats_sha256_not_computed_here": True,
        },
    }
    return _jsonable(result)


def write_events_csv(path: Path, events: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode_index",
        "start_frame",
        "target_frame",
        "start_timestamp",
        "target_timestamp",
        "clipped_dimensions",
        "clipped_dimension_names",
        "raw_target",
        "anchor_state",
        "target_state",
        "relative_target",
        "z",
        "roundtrip_abs_error",
        "same_frame_action_minus_state",
        "anchor_to_target_state_delta",
        "state_neighbor_continuity",
        "action_neighbor_continuity",
        "video_paths",
        "source_parquet",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            row = dict(event)
            for field in fields:
                if isinstance(row.get(field), (list, dict)):
                    row[field] = json.dumps(row[field], separators=(",", ":"), ensure_ascii=True)
            writer.writerow({field: row.get(field, "") for field in fields})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--horizons", default="1,4,8,16,32")
    parser.add_argument("--clip-threshold", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset = args.dataset.expanduser().resolve()
    stats_path = args.stats.expanduser().resolve()
    raw_stats, stats = load_global_stats(stats_path)
    episodes = _read_episode_tables(dataset)
    result = audit_episodes(
        episodes,
        stats,
        dataset,
        horizon=args.horizon,
        horizons=tuple(int(value) for value in args.horizons.split(",") if value.strip()),
        clip_threshold=args.clip_threshold,
        top_k=args.top_k,
    )
    result["provenance"]["stats_sha256"] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    result["provenance"]["stats_source_keys"] = sorted(raw_stats.keys())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if args.output_csv:
        write_events_csv(args.output_csv, result["baseline_clip_events"])
    summary = {
        "output_json": str(args.output_json),
        "output_csv": str(args.output_csv) if args.output_csv else None,
        "episodes": result["dataset_info"]["episodes"],
        "frames": result["dataset_info"]["frames"],
        "baseline_horizon": args.horizon,
        "baseline_clipped_events": len(result["baseline_clip_events"]),
        "baseline_top_windows": len(result["baseline_clip_windows_top"]),
        "state_clip_frames": result["state_clipping"]["frames_with_any_clip"],
        "action_clip_dimensions": result["action_horizon_dependence"][str(args.horizon)]["clipped_dimensions"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
