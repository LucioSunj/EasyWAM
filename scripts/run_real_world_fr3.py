"""Run the FR3 conversion, Robot-backed ActionDiT, text cache and training stages."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from utils.config_resolvers import register_default_resolvers

ROOT = Path(__file__).resolve().parents[1]
TASK = "real_world_fr3_fastwam_idm_wan22"
RAW_NAME = "fr3_example"
STAGES = ("all", "prepare", "convert", "init-action", "text-cache", "train", "config")


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def resolve_workflow(overrides: list[str]):
    """Compose one final configuration shared by every subprocess."""
    register_default_resolvers()
    checkpoint_root = _env_path(
        "FR3_CHECKPOINT_ROOT", ROOT.parent.parent / "checkpoints"
    )
    wan_base = checkpoint_root / "checkpoints/Wan2.2-TI2V-5B"
    robot = _env_path(
        "WAN22_MODEL_DIR", checkpoint_root / "checkpoints/Wan2.2-5B-Robot"
    )
    run_id = os.environ.get("RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    paths = {
        "data.train.dataset_dirs": [
            str(_env_path("FR3_DATA_DIR", ROOT / "data/real_world_fr3_10hz"))
        ],
        "data.train.text_embedding_cache_dir": str(
            _env_path(
                "FR3_TEXT_CACHE", ROOT / "data/text_embeds_cache/real_world_fr3/wan22"
            )
        ),
        "model.backbone.model_id": str(robot),
        "model.backbone.dit_path": str(
            _env_path("WAN22_DIT_PATH", robot / "checkpoint.safetensors")
        ),
        "model.backbone.text_encoder_path": str(
            _env_path("WAN22_T5_PATH", wan_base / "models_t5_umt5-xxl-enc-bf16.pth")
        ),
        "model.backbone.vae_path": str(
            _env_path("WAN22_VAE_PATH", wan_base / "Wan2.2_VAE.pth")
        ),
        "model.backbone.tokenizer_model_id": str(
            _env_path("WAN22_TOKENIZER_DIR", wan_base / "google/umt5-xxl")
        ),
        "model.backbone.action_dit_pretrained_path": str(
            _env_path(
                "FR3_ACTION_DIT_PATH",
                ROOT / "checkpoints/ActionDiT_FR3_Wan22_Robot_alphascale_1024hdim.pt",
            )
        ),
        "output_dir": str(_env_path("FR3_OUTPUT_DIR", ROOT / "runs" / TASK / run_id)),
    }
    common = [
        f"{key}={value if isinstance(value, str) else json.dumps(value)}"
        for key, value in paths.items()
    ]
    common += [
        f"++text_encoder_device={os.environ.get('FR3_TEXT_DEVICE', 'cpu')}",
        f"++text_encoder_dtype={os.environ.get('FR3_TEXT_DTYPE', 'bfloat16')}",
        "++overwrite=false",
        *overrides,
    ]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}", *common])
    OmegaConf.resolve(cfg)
    return cfg, common


def _run(command: list[str], *, env: dict | None = None):
    print(f"[run] {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _require_assets(cfg, *, text: bool = False):
    backbone = cfg.model.backbone
    names = ("text_encoder_path",) if text else ("dit_path", "vae_path")
    for name in names:
        path = Path(str(backbone[name])).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Missing model.backbone.{name}: {path}")
    if text and not Path(str(backbone.tokenizer_model_id)).is_dir():
        raise FileNotFoundError(f"Missing tokenizer: {backbone.tokenizer_model_id}")


def convert_data(cfg, raw_root: Path):
    """Reuse only a complete conversion of the selected raw episode set."""
    roots = list(cfg.data.train.dataset_dirs)
    if len(roots) != 1:
        raise ValueError(
            "The conversion stage requires exactly one output dataset directory."
        )
    output = Path(roots[0]).expanduser().resolve()
    raw_files = sorted((raw_root / "data").glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No raw data/*.parquet under {raw_root}")
    if output.exists():
        summary_path = output / "conversion_summary.json"
        if not summary_path.is_file() or not (output / "meta/info.json").is_file():
            raise ValueError(
                f"Existing output is not a completed FR3 conversion: {output}. Use a new output, or the text-cache/train stages for an existing OpenPI dataset."
            )
        summary = json.loads(summary_path.read_text())
        expected_files = [str(path.resolve()) for path in raw_files]
        actual_files = [row["source_parquet"] for row in summary["episodes"]]
        if (
            summary["input_roots"] != [str(raw_root)]
            or summary["output_fps"] != 10
            or actual_files != expected_files
        ):
            raise ValueError(
                f"Existing conversion does not match all selected raw episodes at 10 Hz: {output}. Choose a new FR3_DATA_DIR."
            )
        print(f"[skip] Complete 10 Hz conversion: {output}", flush=True)
        return
    _run(
        [
            sys.executable,
            "scripts/convert_real_world_fr3.py",
            "--input-roots",
            str(raw_root),
            "--output-root",
            str(output),
            "--fps",
            "10",
        ]
    )


def initialize_action(cfg, common: list[str]):
    """Use native interpolation and alpha scaling from the selected Robot DiT."""
    output = Path(str(cfg.model.action_dit_pretrained_path)).expanduser()
    if output.is_file():
        print(f"[skip] Existing ActionDiT initialization: {output}", flush=True)
        return
    command = [
        sys.executable,
        "scripts/preprocess_action_dit_backbone.py",
        "--task",
        TASK,
        "--backbone",
        "wan22",
        "--output",
        str(output),
        "--device",
        os.environ.get("FR3_PREPROCESS_DEVICE", "cpu"),
        "--dtype",
        "bfloat16",
        "--apply-alpha-scaling",
        "true",
    ]
    for override in common:
        command.extend(["--override", override])
    _run(command)


def run_workflow(stage: str, overrides: list[str]):
    """Run ordered native stages; a failed subprocess prevents training."""
    cfg, common = resolve_workflow(overrides)
    if stage == "config":
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    raw_root = _env_path(
        "FR3_RAW_ROOT", ROOT.parent.parent / "datasets/dataset_real_world" / RAW_NAME
    )
    sequence = {
        "all": ("convert", "init-action", "text-cache", "train"),
        "prepare": ("convert", "init-action", "text-cache"),
    }.get(stage, (stage,))
    # Catch missing local assets before spending time converting all recordings.
    if any(item in sequence for item in ("init-action", "train")):
        _require_assets(cfg)
    if "text-cache" in sequence:
        _require_assets(cfg, text=True)
    for item in sequence:
        print(f"[stage] {item}", flush=True)
        if item == "convert":
            convert_data(cfg, raw_root)
        elif item == "init-action":
            initialize_action(cfg, common)
        elif item == "text-cache":
            _run(
                [
                    sys.executable,
                    "scripts/precompute_text_embeds.py",
                    f"task={TASK}",
                    *common,
                ]
            )
        else:
            if not Path(str(cfg.model.action_dit_pretrained_path)).is_file():
                raise FileNotFoundError(
                    f"Missing ActionDiT initialization: {cfg.model.action_dit_pretrained_path}; run prepare first."
                )
            env = os.environ.copy()
            env["NPROC_PER_NODE"] = env.get(
                "NPROC_PER_NODE", env.get("PET_NPROC_PER_NODE", "1")
            )
            _run(["bash", "scripts/train_zero1.sh", f"task={TASK}", *common], env=env)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Trailing arguments are Hydra overrides shared by every stage. Default all starts training; prepare stops after asset/cache preparation. Run preparation once before a multi-node launch.",
    )
    parser.add_argument("stage", nargs="?", choices=STAGES, default="all")
    args, overrides = parser.parse_known_args()
    run_workflow(args.stage, overrides)


if __name__ == "__main__":
    main()
