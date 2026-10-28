"""Exercise the complete workflow with real Hydra config and subprocess doubles."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fr3_workflow", ROOT / "scripts/run_real_world_fr3.py"
)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    raw = tmp_path / "raw recordings"
    (raw / "data").mkdir(parents=True)
    for index in range(2):
        (raw / "data" / f"episode_{index:06d}.parquet").touch()
    assets = tmp_path / "assets"
    robot = assets / "checkpoints/Wan2.2-5B-Robot"
    base = assets / "checkpoints/Wan2.2-TI2V-5B"
    robot.mkdir(parents=True)
    (base / "google/umt5-xxl").mkdir(parents=True)
    (robot / "checkpoint.safetensors").touch()
    (base / "Wan2.2_VAE.pth").touch()
    (base / "models_t5_umt5-xxl-enc-bf16.pth").touch()
    for name in (
        "WAN22_MODEL_DIR",
        "WAN22_DIT_PATH",
        "WAN22_T5_PATH",
        "WAN22_VAE_PATH",
        "WAN22_TOKENIZER_DIR",
        "NPROC_PER_NODE",
        "PET_NPROC_PER_NODE",
    ):
        monkeypatch.delenv(name, raising=False)
    env = {
        "FR3_CHECKPOINT_ROOT": str(assets),
        "FR3_RAW_ROOT": str(raw),
        "FR3_DATA_DIR": str(tmp_path / "converted"),
        "FR3_TEXT_CACHE": str(tmp_path / "text cache"),
        "FR3_ACTION_DIT_PATH": str(tmp_path / "action.pt"),
        "FR3_OUTPUT_DIR": str(tmp_path / "train output"),
        "CUDA_VISIBLE_DEVICES": "allocated-devices",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return raw, env


def complete_conversion(raw, output):
    output = Path(output)
    (output / "meta").mkdir(parents=True)
    (output / "meta/info.json").write_text('{"fps":10}')
    summary = {
        "input_roots": [str(raw)],
        "output_fps": 10,
        "episodes": [
            {"source_parquet": str(path)}
            for path in sorted((raw / "data").glob("*.parquet"))
        ],
    }
    (output / "conversion_summary.json").write_text(json.dumps(summary))


def test_all_stages_use_final_overrides_and_reuse_preparation(
    workspace, tmp_path, monkeypatch
):
    raw, env = workspace
    output = tmp_path / "chosen data"
    action = tmp_path / "chosen action.pt"
    cache = tmp_path / "chosen cache"
    overrides = [
        f'data.train.dataset_dirs=["{output}"]',
        f'model.backbone.action_dit_pretrained_path="{action}"',
        f'data.train.text_embedding_cache_dir="{cache}"',
        "batch_size=2",
    ]
    calls = []

    def fake_run(command, *, env=None):
        calls.append((command, env))
        if command[1] == "scripts/convert_real_world_fr3.py":
            complete_conversion(raw, command[command.index("--output-root") + 1])
        elif command[1] == "scripts/preprocess_action_dit_backbone.py":
            Path(command[command.index("--output") + 1]).touch()

    monkeypatch.setattr(workflow, "_run", fake_run)
    workflow.run_workflow("all", overrides)
    assert [command[1] for command, _ in calls] == [
        "scripts/convert_real_world_fr3.py",
        "scripts/preprocess_action_dit_backbone.py",
        "scripts/precompute_text_embeds.py",
        "scripts/train_zero1.sh",
    ]
    assert calls[0][0][calls[0][0].index("--output-root") + 1] == str(output)
    assert calls[1][0][calls[1][0].index("--output") + 1] == str(action)
    action_overrides = [
        calls[1][0][index + 1]
        for index, arg in enumerate(calls[1][0])
        if arg == "--override"
    ]
    assert action_overrides == calls[2][0][3:] == calls[3][0][3:]
    assert action_overrides[-len(overrides) :] == overrides
    assert calls[3][1]["CUDA_VISIBLE_DEVICES"] == "allocated-devices"
    assert calls[3][1]["NPROC_PER_NODE"] == "1"
    assert not Path(env["FR3_DATA_DIR"]).exists()
    calls.clear()
    workflow.run_workflow("all", overrides)
    assert [command[1] for command, _ in calls] == [
        "scripts/precompute_text_embeds.py",
        "scripts/train_zero1.sh",
    ]


@pytest.mark.parametrize(
    "incomplete", [True, False], ids=["missing_summary", "episode_subset"]
)
def test_existing_conversion_must_cover_current_raw_set(
    workspace, monkeypatch, incomplete
):
    raw, env = workspace
    output = Path(env["FR3_DATA_DIR"])
    complete_conversion(raw, output)
    summary_path = output / "conversion_summary.json"
    if incomplete:
        summary_path.unlink()
    else:
        summary = json.loads(summary_path.read_text())
        summary["episodes"].pop()
        summary_path.write_text(json.dumps(summary))
    calls = []
    monkeypatch.setattr(
        workflow, "_run", lambda command, **kwargs: calls.append(command)
    )
    with pytest.raises(ValueError, match="Existing"):
        workflow.run_workflow("all", [])
    assert calls == []
    assert output.exists() and len(list((raw / "data").glob("*.parquet"))) == 2


def test_failed_preparation_never_launches_training(workspace, monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(workflow, "_run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        workflow.run_workflow("all", [])
    assert len(calls) == 1 and calls[0][1] == "scripts/convert_real_world_fr3.py"


def test_prepare_stops_before_training(workspace, monkeypatch):
    raw, env = workspace
    complete_conversion(raw, env["FR3_DATA_DIR"])
    Path(env["FR3_ACTION_DIT_PATH"]).touch()
    calls = []
    monkeypatch.setattr(
        workflow, "_run", lambda command, **kwargs: calls.append(command)
    )
    workflow.run_workflow("prepare", [])
    assert [command[1] for command in calls] == ["scripts/precompute_text_embeds.py"]
