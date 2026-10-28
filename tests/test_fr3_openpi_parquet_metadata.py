"""Read OpenPI's newer HF List metadata through the native v2.1 loader."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from data.lerobot.lerobot.lerobot_dataset import LeRobotDataset


def write_openpi_list_dataset(root):
    """Create valid episodes, then reproduce the supplied OpenPI schema metadata."""
    writer = LeRobotDataset.create(
        repo_id="local/fr3_metadata",
        root=root,
        fps=10,
        robot_type="fr3",
        use_videos=False,
        video_backend="pyav",
        features={
            "observation.state": {"dtype": "float32", "shape": (8,), "names": ["qpos"]},
            "action": {"dtype": "float32", "shape": (8,), "names": ["action"]},
        },
    )
    expected = []
    for episode, length in enumerate((3, 2)):
        task = f"Insert tube {episode} into the rack."
        for frame in range(length):
            state = np.arange(8, dtype=np.float32) + episode * 10 + frame
            action = state + np.float32(0.25)
            writer.add_frame(
                {"observation.state": state, "action": action},
                task=["", task, "", ""],
            )
            expected.append((state, action, task, episode, frame))
        writer.save_episode()

    source_bytes = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        metadata = dict(table.schema.metadata)
        hf_metadata = json.loads(metadata[b"huggingface"])
        for key in ("observation.state", "action"):
            hf_metadata["info"]["features"][key]["_type"] = "List"
        metadata[b"huggingface"] = json.dumps(hf_metadata).encode("utf-8")
        pq.write_table(table.replace_schema_metadata(metadata), path)
        source_bytes[path] = path.read_bytes()
    return expected, source_bytes


@pytest.mark.parametrize(
    "episodes", [None, [0]], ids=["all_episodes", "selected_episode"]
)
def test_v21_loads_openpi_list_metadata_without_rewriting_source(tmp_path, episodes):
    root = tmp_path / "openpi"
    expected, source_bytes = write_openpi_list_dataset(root)
    dataset = LeRobotDataset(
        repo_id="local/fr3_metadata",
        root=root,
        episodes=episodes,
        video_backend="pyav",
    )
    if episodes is not None:
        expected = [row for row in expected if row[3] in episodes]
    assert len(dataset) == len(expected)
    for index, (state, action, task, episode, frame) in enumerate(expected):
        sample = dataset[index]
        torch.testing.assert_close(
            sample["observation.state"], torch.from_numpy(state), atol=0, rtol=0
        )
        torch.testing.assert_close(
            sample["action"], torch.from_numpy(action), atol=0, rtol=0
        )
        assert sample["task"] == task
        assert sample["episode_index"].item() == episode
        assert sample["frame_index"].item() == frame
        assert sample["index"].item() == index
        torch.testing.assert_close(
            sample["timestamp"], torch.tensor(frame / 10), atol=0, rtol=0
        )
    assert all(path.read_bytes() == original for path, original in source_bytes.items())
