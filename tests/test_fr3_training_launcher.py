"""Check real config composition and mock the GPU-facing launcher commands."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_real_world_fr3.sh"


def launcher_env(tmp_path):
    """Isolate test paths while preserving the interpreter's Python environment."""
    env = os.environ.copy()
    for name in (
        "NPROC_PER_NODE",
        "PET_NPROC_PER_NODE",
        "NNODES",
        "PET_NNODES",
        "NODE_RANK",
        "PET_NODE_RANK",
        "MASTER_ADDR",
        "PET_MASTER_ADDR",
        "MASTER_PORT",
        "PET_MASTER_PORT",
        "FR3_DATA_DIR",
    ):
        env.pop(name, None)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env.update(
        PATH=f"{bin_dir}{os.pathsep}{env['PATH']}",
        CUDA_VISIBLE_DEVICES="",
        FR3_TEXT_CACHE=str(tmp_path / "text cache"),
        WAN22_MODEL_DIR=str(tmp_path / "wan weights"),
        FR3_ACTION_DIT_PATH=str(tmp_path / "action backbone.pt"),
        FR3_OUTPUT_DIR=str(tmp_path / "default output"),
        RUN_ID="fr3_launcher_test",
        FR3_TEST_RECORD=str(tmp_path / "invocation.json"),
    )
    return env, bin_dir


def command_recorder(path):
    """Record dispatch instead of loading the text encoder or starting training."""
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "record = {'args': sys.argv[1:], 'cwd': os.getcwd(), "
        "'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'pythonpath': os.environ.get('PYTHONPATH')}\n"
        "Path(os.environ['FR3_TEST_RECORD']).write_text(json.dumps(record))\n"
        "print('mock command completed', flush=True)\n"
    )
    path.chmod(0o755)


def test_config_stage_resolves_fr3_without_creating_training_output(tmp_path):
    env, bin_dir = launcher_env(tmp_path)
    python = bin_dir / "python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "config",
            "batch_size=2",
            "gradient_accumulation_steps=8",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    cfg = OmegaConf.create(result.stdout)
    assert cfg.model._target_ == "runtime.create_fastwam_idm"
    assert cfg.model.action_dit_config.action_dim == cfg.model.state_dim == 8
    assert cfg.model.backbone.model_id == env["WAN22_MODEL_DIR"]
    assert (
        cfg.model.backbone.tokenizer_model_id
        == env["WAN22_MODEL_DIR"] + "/google/umt5-xxl"
    )
    assert cfg.model.action_dit_pretrained_path == env["FR3_ACTION_DIT_PATH"]
    assert cfg.data.train.text_embedding_cache_dir == env["FR3_TEXT_CACHE"]
    assert (
        cfg.data.train.num_frames == 17 and cfg.data.train.action_video_freq_ratio == 4
    )
    expected = (
        ROOT.parent.parent
        / "datasets/dataset_real_world"
        / (
            "fr3_example_conversion_reference_20261011/"
            "converted_dataset/lsk/fr3_example"
        )
    )
    assert Path(cfg.data.train.dataset_dirs[0]).resolve() == expected
    assert cfg.batch_size == 2 and cfg.gradient_accumulation_steps == 8
    assert cfg.max_steps == 20000 and cfg.eval_every == 1000
    assert cfg.resume is None
    assert cfg.output_dir == env["FR3_OUTPUT_DIR"]
    assert not Path(cfg.output_dir).exists()


def test_text_cache_dispatch_shares_dataset_and_assets(tmp_path):
    env, bin_dir = launcher_env(tmp_path)
    env["FR3_DATA_DIR"] = str(tmp_path / "converted data")
    command_recorder(bin_dir / "python")
    subprocess.run(
        ["bash", str(SCRIPT), "text-cache", "++overwrite=true"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    record = json.loads(Path(env["FR3_TEST_RECORD"]).read_text())
    args = record["args"]
    assert args[0] == "scripts/precompute_text_embeds.py"
    assert f"data.train.dataset_dirs=[{env['FR3_DATA_DIR']}]" in args
    assert f"model.backbone.model_id={env['WAN22_MODEL_DIR']}" in args
    assert f"data.train.text_embedding_cache_dir={env['FR3_TEXT_CACHE']}" in args
    assert args[-2:] == ["++overwrite=false", "++overwrite=true"]
    assert Path(record["cwd"]) == ROOT
    assert record["pythonpath"].split(os.pathsep)[0] == str(ROOT / "src")
    assert not Path(env["FR3_OUTPUT_DIR"]).exists()


@pytest.mark.parametrize(
    "processes", [None, "2"], ids=["one_process_default", "scheduler_processes"]
)
def test_train_dispatch_preserves_devices_resume_and_final_output(tmp_path, processes):
    env, bin_dir = launcher_env(tmp_path)
    if processes is not None:
        env["PET_NPROC_PER_NODE"] = processes
    # The mocked command must inherit the user's selection without querying it.
    env["CUDA_VISIBLE_DEVICES"] = "allocated-devices"
    env["FR3_DATA_DIR"] = str(tmp_path / "converted data")
    command_recorder(bin_dir / "accelerate")
    output = tmp_path / "chosen output"
    resume = str(tmp_path / "previous/checkpoints/state/step_002000")
    subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "train",
            f"output_dir={output}",
            f"resume={resume}",
            "batch_size=2",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    record = json.loads(Path(env["FR3_TEST_RECORD"]).read_text())
    args = record["args"]
    assert args[0] == "launch"
    assert args[args.index("--num_processes") + 1] == (processes or "1")
    assert "scripts/train.py" in args
    assert "task=real_world_fr3_fastwam_idm_wan22" in args
    assert f"data.train.dataset_dirs=[{env['FR3_DATA_DIR']}]" in args
    assert f"resume={resume}" in args and "batch_size=2" in args
    assert [arg for arg in args if arg.startswith("output_dir=")][
        -1
    ] == f"output_dir={output}"
    assert record["cuda_visible_devices"] == "allocated-devices"
    assert Path(record["cwd"]) == ROOT
    assert "mock command completed" in (output / "train.log").read_text()
    assert not Path(env["FR3_OUTPUT_DIR"]).exists()
