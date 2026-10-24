"""Exercise the FR3 converter through real video files and the training loader."""

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from scripts.convert_real_world_fr3 import (
    CAMERA_IDS,
    convert_dataset,
    frame_stride_for_fps,
)


def write_raw_episode(root, episode, count=55, camera_counts=None):
    """Write distinct camera colors and changing joint targets at 30 Hz."""
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "videos").mkdir(exist_ok=True)
    index = np.arange(count)
    state = np.column_stack([0.03 * index[:, None] + np.arange(7) * 0.1, index % 2])
    action = state.copy()
    action[:, :7] += 0.1
    action[:, 7] = 1 - action[:, 7]
    table = pd.DataFrame(
        {
            "timestamp": 1000 + index / 30,
            "qpos": list(state),
            "action": list(action),
            "language_instruction": ["Insert the tubes into the rack one by one."]
            * count,
        }
    )
    stem = f"episode_gello_{episode:06d}"
    table.to_parquet(root / "data" / f"{stem}.parquet")
    for name, serial in CAMERA_IDS.items():
        path = root / "videos" / f"cam_{serial}_{stem}.mp4"
        with av.open(str(path), "w") as output:
            stream = output.add_stream("libx264", rate=30, options={"crf": "0"})
            stream.width, stream.height = 64, 48
            stream.pix_fmt = "yuv444p"
            video_count = count if camera_counts is None else camera_counts[name]
            for frame_index in range(video_count):
                color = (
                    [180, 3 * frame_index, 30]
                    if name == "left"
                    else [30, 3 * frame_index, 180]
                )
                image = np.full((48, 64, 3), color, dtype=np.uint8)
                for packet in stream.encode(
                    av.VideoFrame.from_ndarray(image, format="rgb24")
                ):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    return table


def test_raw_conversion_and_training_sample(tmp_path, monkeypatch):
    raw, output = tmp_path / "raw", tmp_path / "converted"
    original = [write_raw_episode(raw, 0), write_raw_episode(raw, 1, count=10)]
    convert_dataset([raw], output)
    info = json.loads((output / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert (info["fps"], info["total_episodes"], info["total_frames"]) == (10, 2, 23)
    summary = json.loads((output / "conversion_summary.json").read_text())
    assert [item["stride"] for item in summary["episodes"]] == [3, 3]
    tasks = {
        row["task_index"]: row["task"]
        for row in map(
            json.loads, (output / "meta/tasks.jsonl").read_text().splitlines()
        )
    }
    offset = 0
    for episode_index, source in enumerate(original):
        converted = pd.read_parquet(
            output / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        for before, after in (("qpos", "observation.state"), ("action", "action")):
            np.testing.assert_array_equal(
                np.stack(converted[after]),
                np.stack(source[before])[::3].astype(np.float32),
            )
        np.testing.assert_allclose(
            converted.timestamp, np.arange(len(converted)) / 10, atol=1e-6
        )
        np.testing.assert_array_equal(
            converted["index"], np.arange(offset, offset + len(converted))
        )
        assert [
            tasks[index] for index in converted.task_index
        ] == source.language_instruction.iloc[::3].tolist()
        offset += len(converted)
        for camera in CAMERA_IDS:
            path = (
                output
                / f"videos/chunk-000/observation.images.{camera}/episode_{episode_index:06d}.mp4"
            )
            with av.open(str(path)) as video:
                assert video.streams.video[0].average_rate == 10
                frames = [
                    frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)
                ]
            assert len(frames) == len(converted)
            for output_index in (0, len(frames) // 2, len(frames) - 1):
                rgb = frames[output_index].mean((0, 1))
                expected = (
                    [180, 9 * output_index, 30]
                    if camera == "left"
                    else [30, 9 * output_index, 180]
                )
                np.testing.assert_allclose(rgb, expected, atol=6)
    assert not (output / "images").exists()

    # Read through the real dataset/processor, with a synthetic text cache only.
    # No text encoder, VAE, policy, or GPU is constructed here.
    from data.lerobot.prompts import DEFAULT_PROMPT
    from data.lerobot.text_embedding_cache import (
        build_text_embedding_payload,
        prompt_hash,
        text_embedding_cache_filename,
    )
    from utils import misc

    cache = tmp_path / "cache"
    cache.mkdir()
    prompt = DEFAULT_PROMPT.format(task=original[0].language_instruction.iloc[0])
    mask = torch.arange(128) < 3
    torch.save(
        build_text_embedding_payload(
            context=torch.zeros(128, 4096),
            mask=mask,
            context_len=128,
            encoder_id="wan22ti2v5b",
            prompt_digest=prompt_hash(prompt),
        ),
        cache / text_embedding_cache_filename(prompt, 128, "wan22ti2v5b"),
    )
    monkeypatch.setattr(misc, "_WORK_DIR", str(tmp_path))
    config_root = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        cfg = compose(
            config_name="train", overrides=["task=real_world_fr3_fastwam_idm_wan22"]
        )
    cfg.data.train.dataset_dirs = [str(output)]
    cfg.data.train.text_embedding_cache_dir = str(cache)
    dataset = instantiate(cfg.data.train)
    sample = dataset._get(1)
    assert sample["video"].shape == (3, 5, 224, 448)
    assert sample["action"].shape == sample["proprio"].shape == (16, 8)
    assert sample["prompt"] == prompt
    torch.testing.assert_close(sample["context_mask"], mask)
    assert not sample["action_is_pad"].any()
    assert not sample["image_is_pad"].any()
    processor = dataset.lerobot_dataset.processor
    normalized_batch = {
        "action": {"default": sample["action"]},
        "state": {"default": sample["proprio"]},
    }
    restored = processor.normalizer.backward(normalized_batch)
    for transform in reversed(processor.action_state_transforms):
        restored = transform.backward(restored)
    expected_actions = torch.tensor(
        np.stack(original[0].action)[::3][1:17], dtype=torch.float32
    )
    torch.testing.assert_close(
        restored["action"]["default"], expected_actions, atol=2e-6, rtol=2e-6
    )
    # The final window must pad inside episode 0, not take episode 1's start.
    terminal = dataset._get(18)
    assert terminal["action_is_pad"].tolist() == [False] + [True] * 15
    assert terminal["image_is_pad"].tolist() == [False] + [True] * 4
    stats = json.loads((tmp_path / "dataset_stats.json").read_text())
    assert len(stats["action"]["default"]["global_mean"]) == 8
    assert stats["num_episodes"] == 2


def test_conversion_rejects_misaligned_camera_without_publishing_dataset(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "output"
    write_raw_episode(raw, 0, count=10, camera_counts={"left": 10, "wrist": 9})
    with pytest.raises(ValueError, match="9 video frames vs 10 table rows"):
        convert_dataset([raw], output)
    assert not output.exists()
    assert len(list((raw / "videos").glob("*.mp4"))) == 2


def test_existing_output_and_already_converted_input_are_preserved(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing data")
    with pytest.raises(FileExistsError):
        convert_dataset([tmp_path / "raw"], output)
    assert marker.read_text() == "existing data"
    (output / "meta").mkdir()
    (output / "meta/info.json").write_text('{"codebase_version": "v2.1"}')
    with pytest.raises(ValueError, match="already LeRobot"):
        convert_dataset([output], tmp_path / "new")


def test_sampling_rejects_fractional_ratio_and_does_not_resample_10hz():
    assert frame_stride_for_fps(30, 10) == 3
    assert frame_stride_for_fps(10, 10) == 1
    with pytest.raises(ValueError, match="integer ratio"):
        frame_stride_for_fps(25, 10)
