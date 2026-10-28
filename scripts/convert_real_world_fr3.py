"""Convert FR3 Parquet + synchronized left/wrist MP4 recordings to LeRobot v2.1.

The output retains absolute eight-dimensional actions and measured states:
seven joints in radians, then the collector's gripper (0=open, 1=closed).
The real_world_fr3 training processor makes the first seven action dimensions
relative to the current observation after constructing each action chunk.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

CAMERA_IDS = {"left": "000000000001", "wrist": "000000000002"}
REQUIRED_COLUMNS = {"timestamp", "qpos", "action", "language_instruction"}


def frame_stride_for_fps(source_fps: float, output_fps: int) -> int:
    """Return the shared integer sampling stride for all frame modalities."""
    if not np.isfinite(source_fps) or source_fps <= 0 or output_fps <= 0:
        raise ValueError(f"Invalid FPS: source={source_fps}, output={output_fps}")
    ratio = source_fps / output_fps
    stride = round(ratio)
    if stride < 1 or not np.isclose(ratio, stride, rtol=0, atol=1e-3):
        raise ValueError(
            f"FPS must have an integer ratio: {source_fps} -> {output_fps}"
        )
    return stride


def read_episode_table(path: Path) -> pd.DataFrame:
    """Read recorded values without changing action coordinates or gripper sign."""
    table = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing or table.empty:
        raise ValueError(f"{path}: empty episode or missing columns {sorted(missing)}")
    for key in ("qpos", "action"):
        values = np.asarray(table[key].tolist(), dtype=np.float32)
        if values.shape != (len(table), 8) or not np.isfinite(values).all():
            raise ValueError(
                f"{path}: {key} must contain finite eight-dimensional vectors"
            )
        table[key] = list(values)
    if any(
        not isinstance(text, str) or not text.strip()
        for text in table.language_instruction
    ):
        raise ValueError(f"{path}: every frame needs a nonempty language_instruction")
    return table


def video_paths(parquet_path: Path) -> dict[str, Path]:
    """Resolve the camera serials used by the supplied FR3 recording pipeline."""
    directory = parquet_path.parent.parent / "videos"
    return {
        name: directory / f"cam_{serial}_{parquet_path.stem}.mp4"
        for name, serial in CAMERA_IDS.items()
    }


def build_features(image_shape: tuple[int, int, int]) -> dict:
    """Describe RGB video and unnormalized joint/gripper vectors on disk."""
    return {
        **{
            f"observation.images.{name}": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            }
            for name in CAMERA_IDS
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["qpos"]},
        "action": {"dtype": "float32", "shape": (8,), "names": ["action"]},
    }


def convert_episode(
    dataset, parquet_path: Path, image_shape: tuple[int, int, int]
) -> dict:
    """Decode the same source indices for both cameras and numerical fields."""
    table = read_episode_table(parquet_path)
    paths = video_paths(parquet_path)
    captures = {}
    try:
        for name, path in paths.items():
            capture = cv2.VideoCapture(str(path))
            captures[name] = capture
            if not capture.isOpened():
                raise FileNotFoundError(f"Cannot open {name} video: {path}")
        source_fps = captures["left"].get(cv2.CAP_PROP_FPS)
        stride = frame_stride_for_fps(source_fps, dataset.fps)
        for name, capture in captures.items():
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not np.isclose(fps, source_fps, rtol=0, atol=1e-3):
                raise ValueError(
                    f"{paths[name]}: camera FPS differs ({fps} vs {source_fps})"
                )
            if count != len(table):
                raise ValueError(
                    f"{paths[name]}: {count} video frames vs {len(table)} table rows"
                )

        episode_index = dataset.meta.total_episodes
        written = 0
        for source_index, row in enumerate(table.itertuples(index=False)):
            selected = source_index % stride == 0
            frame = {}
            for name, capture in captures.items():
                if not capture.grab():
                    raise ValueError(
                        f"{paths[name]}: missing source frame {source_index}"
                    )
                if not selected:
                    continue
                ok, bgr = capture.retrieve()
                if not ok:
                    raise ValueError(
                        f"{paths[name]}: cannot decode source frame {source_index}"
                    )
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape != image_shape:
                    raise ValueError(
                        f"{paths[name]}: image shape {rgb.shape}, expected {image_shape}"
                    )
                key = f"observation.images.{name}"
                # The vendored writer only stages paths. Its encoder scans the
                # .jpeg suffix, but PIL reads image headers: use lossless PNG
                # contents for these temporary files to avoid a JPEG encode.
                path = dataset._get_image_file_path(episode_index, key, written)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(path, format="PNG")
                frame[key] = rgb
            if selected:
                frame.update({"observation.state": row.qpos, "action": row.action})
                # Only the low-level instruction is present in these recordings.
                dataset.add_frame(frame, task=["", row.language_instruction, "", ""])
                written += 1
        dataset.save_episode(raw_file_name=str(parquet_path))
        return {
            "episode_index": episode_index,
            "source_parquet": str(parquet_path),
            "source_videos": {name: str(path) for name, path in paths.items()},
            "source_fps": source_fps,
            "source_frames": len(table),
            "stride": stride,
            "output_frames": written,
        }
    finally:
        for capture in captures.values():
            capture.release()


def convert_dataset(
    input_roots: list[Path],
    output_root: Path,
    *,
    fps: int = 10,
    max_episodes: int | None = None,
) -> Path:
    """Write a new local dataset, leaving raw recordings and existing outputs intact."""
    if fps <= 0 or (max_episodes is not None and max_episodes <= 0):
        raise ValueError("fps and max_episodes must be positive")
    roots = [path.expanduser().resolve() for path in input_roots]
    output = output_root.expanduser().resolve()
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Provide distinct input roots")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    episodes = []
    for root in roots:
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("Input and output directories must be independent")
        if (root / "meta/info.json").exists():
            raise ValueError(
                f"{root} is already LeRobot data; use it directly without resampling"
            )
        files = sorted((root / "data").glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No raw data/*.parquet in {root}")
        episodes.extend(files)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    read_episode_table(episodes[0])
    first_video = video_paths(episodes[0])["left"]
    capture = cv2.VideoCapture(str(first_video))
    try:
        ok, first_frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot read first frame: {first_video}")
        image_shape = first_frame.shape
    finally:
        capture.release()

    from data.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "dataset"
        dataset = LeRobotDataset.create(
            repo_id=f"local/{output.name}",
            root=staged,
            fps=fps,
            robot_type="fr3",
            features=build_features(image_shape),
            video_codec="h264",
        )
        records = []
        try:
            for path in episodes:
                record = convert_episode(dataset, path, image_shape)
                records.append(record)
                print(
                    f"{path.name}: {record['source_frames']} -> {record['output_frames']} frames "
                    f"({record['source_fps']:g} -> {fps} Hz)",
                    flush=True,
                )
        finally:
            dataset.stop_image_writer()
        summary = {
            "input_roots": [str(root) for root in roots],
            "output_fps": fps,
            "camera_ids": CAMERA_IDS,
            "image_shape": list(image_shape),
            "alignment": "shared_source_frame_index",
            "action_representation": "absolute_joint_targets_7_plus_unchanged_gripper_1",
            "total_episodes": len(records),
            "total_frames": sum(record["output_frames"] for record in records),
            "episodes": records,
        }
        (staged / "conversion_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staged.rename(output)
    return output


def main() -> None:
    """Parse the raw recording conversion command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--max-episodes", type=int, help="Convert only the first N sorted episodes"
    )
    args = parser.parse_args()
    path = convert_dataset(
        args.input_roots, args.output_root, fps=args.fps, max_episodes=args.max_episodes
    )
    print(f"Completed: {path}")


if __name__ == "__main__":
    main()
