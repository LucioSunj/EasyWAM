import numpy as np
from pathlib import Path

from scripts.audit_fr3_clipping import (
    _contiguous_runs,
    audit_episodes,
    normalize_and_error,
)


def _episode(episode_index, state, action):
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    frames = np.arange(len(state), dtype=np.int64)
    return {
        "episode_index": episode_index,
        "frame": frames,
        "timestamp": frames.astype(np.float64) / 10.0,
        "state": state,
        "action": action,
        "path": f"/synthetic/data/episode_{episode_index:06d}.parquet",
    }


def _stats():
    zeros = np.zeros(8, dtype=np.float64)
    ones = np.ones(8, dtype=np.float64)
    return {"state": {"mean": zeros, "std": ones}, "action": {"mean": zeros, "std": ones}}


def test_clamp_reports_raw_z_and_roundtrip_error():
    values = np.asarray([[6.0, -7.0] + [0.0] * 6])
    z, clipped, error = normalize_and_error(values, np.zeros(8), np.ones(8), 5.0)
    assert np.allclose(z[0, :2], [6.0, -7.0])
    assert clipped[0, :2].tolist() == [True, True]
    assert np.allclose(error[0, :2], [1.0, 2.0])


def test_horizon_relative_anchor_and_exact_event_position():
    # The target at frame 2 equals its same-frame state, so H=1 has no clip.
    # It is six units away from the frame-1 anchor, so H=2 has one exact clip.
    state = np.zeros((4, 8), dtype=np.float64)
    action = np.zeros((4, 8), dtype=np.float64)
    state[2:, 0] = 6.0
    action[2:, 0] = 6.0
    result = audit_episodes(
        [_episode(0, state, action)],
        _stats(),
        dataset_root=Path("/synthetic"),
        horizon=2,
        horizons=(1, 2),
        top_k=5,
    )

    h1 = result["action_horizon_dependence"]["1"]
    h2 = result["action_horizon_dependence"]["2"]
    assert h1["clipped_vectors"] == 0
    assert h2["clipped_vectors"] == 1
    assert h2["clipped_dimensions"] == [1, 0, 0, 0, 0, 0, 0, 0]
    assert len(result["baseline_clip_events"]) == 1
    event = result["baseline_clip_events"][0]
    assert (event["episode_index"], event["start_frame"], event["target_frame"]) == (0, 1, 2)
    assert event["clipped_dimension_names"] == ["joint_0"]
    assert event["same_frame_action_minus_state"][0] == 0.0


def test_contiguous_runs_are_deterministic_and_deduplicated():
    assert _contiguous_runs([5, 3, 4, 4, 9]) == [
        {"start_frame": 3, "end_frame": 5, "length": 3},
        {"start_frame": 9, "end_frame": 9, "length": 1},
    ]
