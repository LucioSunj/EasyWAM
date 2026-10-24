# FR3 Real-World Data for FastWAM-IDM

[中文](real_world_fr3_zh.md) | [Data index](README.md) | [FastWAM configuration](../config/fastwam.md)

The `real_world_fr3_fastwam_idm_wan22` task trains EasyWAM's Wan2.2 FastWAM-IDM on FR3 recordings with a left camera, a wrist camera, seven joint positions, and one gripper value. It retains 10 Hz data and 16-action chunks. This is a supervised data entrypoint; robot deployment and the LIBERO BC/RL handoff are separate integrations.

Run the following commands from `EasyWAM/`. The one-command shell entrypoint selects the prepared `.venv` and sets the source import path without manual activation; other standalone Python scripts still require the project environment.

## Local training environment

The local environment is `EasyWAM/.venv`, created with Python 3.10.20 from the existing `conda/envs/troncamp_env`. Its `include-system-site-packages=false` setting isolates the packages without changing the source Conda environment. Activate it before use:

```bash
cd /path/to/storage/Outer/EasyWAM
source .venv/bin/activate
bash scripts/run_real_world_fr3.sh config
```

Dependencies follow the exact `pyproject.toml` versions, including PyTorch `2.7.1+cu128`, Torchvision `0.22.1+cu128`, Accelerate `1.5.2`, DeepSpeed `0.16.9`, Transformers `4.57.0`, Datasets `3.6.0`, and PyAV `16.0.1`. EasyWAM is installed in editable mode, with `pytest` added for CPU regression tests. FlashAttention is not installed; the default `attention_backend=auto` uses the supported PyTorch SDPA fallback.

The environment includes CUDA 12.8 runtime libraries, but no separately installed CUDA compiler or precompiled DeepSpeed extensions. The current ZeRO-1 recipe uses PyTorch fused AdamW with optimizer offload disabled and does not require every DeepSpeed custom operator. Features that compile extensions need a matching CUDA toolkit; the host's `/usr/local/cuda` points to 12.0 and is not a configured 12.8 compiler environment.

To recreate the environment, use the existing `uv` installation and a Python 3.10 interpreter. Configure network access according to the cluster instructions, then run from EasyWAM:

```bash
uv venv --python /absolute/path/to/python3.10 .venv
uv pip install --python .venv/bin/python pip setuptools wheel ninja pytest
DS_BUILD_OPS=0 UV_LINK_MODE=copy uv pip install --python .venv/bin/python \
  --index https://pypi.org/simple \
  --default-index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match -e .
.venv/bin/python -m pip check
```

This index order prefers PyPI for ordinary packages at the same version; CUDA Torch/Torchvision come from PyTorch. The first attempt with reversed priority directed NCCL to `pypi.nvidia.com`, which timed out on this node; its failure log is preserved. Do not rerun environment creation over an existing `.venv`. Installation does not download Wan2.2 / ActionDiT weights, generate real text embeddings, or start training; those steps are described below.

The independent environment passed acceptance on 2026-10-11: `pip check` found no dependency conflicts; the FastWAM/Wan, trainer, and FR3 data imports succeeded; CPU SDPA forward/backward, the PyAV H.264 encoder, and configuration parsing passed; and **16 FR3 regression tests passed** in 189.77 seconds with one torchvision deprecation warning. Exact commands, versions, the installation retry, and check logs are recorded in the [environment acceptance report](../../../../test_artifacts/easywam_training_env_20261011_v1/RESULTS.json), with a separate [package inventory](../../../../test_artifacts/easywam_training_env_20261011_v1/packages.json). The `python`, `accelerate`, `hf`, and `ninja` entrypoints all resolve under `.venv/bin`.

At environment acceptance, this node had no mounted NVIDIA devices, and the older entrypoint's default Wan2.2 and ActionDiT paths did not exist. The new workflow now selects the existing Robot DiT, UMT5, VAE, and tokenizer in the parent checkpoint directory, and real text embeddings have subsequently been generated on CPU as described below. CUDA/NCCL, GPU memory use, training, and robot execution remain **NOT-RUN**. The CPU checks used command-scoped `DS_ACCELERATOR=cpu` and empty `CUDA_VISIBLE_DEVICES`; neither setting was persisted in activation scripts or training defaults.

For a source checkout without an editable EasyWAM installation, expose `src`
before running the scripts:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Choose the input dataset

An existing OpenPI conversion can be used directly when its LeRobot v2.1 dataset contains these fields:

| Field | Stored representation |
| --- | --- |
| `observation.images.left` | RGB video, 640 × 480, 10 FPS |
| `observation.images.wrist` | RGB video, 640 × 480, 10 FPS |
| `observation.state` | `float32[8]`: seven absolute joint positions, then gripper state |
| `action` | `float32[8]`: seven absolute joint targets, then the original gripper command |
| Task | Original instruction through `task_index` and `meta/tasks.jsonl` |

The reference Parquet files carry Hugging Face `List` feature metadata that EasyWAM's `datasets==3.6.0` cannot deserialize automatically. The v2.1 loader now derives the read schema from LeRobot's `meta/info.json` features for both whole-corpus and selected-episode reads. This leaves the source Parquet files and MP4 videos unchanged; v3 loading is unchanged. Both metadata regression cases and direct loading of reference episode 0 pass.

The supplied reference dataset is already converted:

```bash
OPENPI_REFERENCE=/path/to/storage/datasets/dataset_real_world/fr3_example_conversion_reference_20261011/converted_dataset/lsk/fr3_example
```

Its metadata records 100 episodes, 52,529 frames, 200 videos, and one task at 10 Hz. It is the same collection as the neighboring raw `fr3_example` directory. Choose one representation; including both would duplicate demonstrations. Do not downsample this 10 Hz dataset again or modify the reference package.

The new `run_real_world_fr3.py` workflow and direct Hydra composition both default to `./data/real_world_fr3_10hz` as the raw-conversion output. Only the older `train_real_world_fr3.sh` defaults to the existing reference dataset above. Use `FR3_DATA_DIR` or `data.train.dataset_dirs` for another location. Select `text-cache` and `train` to use an existing OpenPI conversion; do not select it as the conversion output of `all` or `prepare`. The dataset layout is:

```text
<dataset>/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/
│   ├── observation.images.left/episode_000000.mp4
│   └── observation.images.wrist/episode_000000.mp4
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    └── tasks.jsonl
```

## Convert raw recordings when needed

`scripts/convert_real_world_fr3.py` reads raw episodes in this layout:

```text
<raw-root>/
├── data/episode_gello_000000.parquet
└── videos/
    ├── cam_000000000001_episode_gello_000000.mp4
    └── cam_000000000002_episode_gello_000000.mp4
```

The camera mapping is fixed: `000000000001 → left`, `000000000002 → wrist`. Required Parquet columns are `timestamp`, `qpos`, `action`, and `language_instruction`. Each row's state and action must have eight finite values. Both videos must match the episode's row count and source frame rate.

```bash
python scripts/convert_real_world_fr3.py \
  --input-roots /path/to/raw/fr3_example \
  --output-root ./data/real_world_fr3_10hz \
  --fps 10
```

`--input-roots` accepts multiple raw roots. The output directory must be new. All episodes are converted by default; `--max-episodes 1` creates a separate small conversion for inspection. The script writes local LeRobot v2.1 data and a `conversion_summary.json` recording source paths, camera mapping, frame rates, and per-episode source/output frame counts. It does not upload data or overwrite the sources.

For the supplied 30 FPS recordings, conversion selects source indices `0, 3, 6, ...` for both cameras, `qpos`, `action`, and instruction together. Episode length becomes `ceil(source_frames / 3)`; output frame index `k` has timestamp `k / 10`. Alignment follows frame indices rather than interpolation of raw timestamps, so conversion does not correct acquisition delays already present in the recording. Source/output frame rates must admit an integer stride.

Images are decoded from BGR to RGB and written as H.264 videos using the native writer. Their original orientation and 640 × 480 field of view are preserved; video re-encoding is lossy. States and actions are stored as absolute `float32[8]` values. Conversion does not subtract joints, normalize values, binarize/invert the gripper, or apply LIBERO's image rotation. The recording code uses radians for the first seven joint dimensions and `0=open, 1=closed` for the final dimension; conversion preserves those recorded values.

## What the training loader constructs

The recipe is in `configs/data/real_world_fr3.yaml`, selected by `configs/task/real_world_fr3_fastwam_idm_wan22.yaml`.

| Setting | Value and effect |
| --- | --- |
| `num_frames` | `17` observations, including the current observation |
| `global_sample_stride` | `1`; use the stored 10 Hz timeline |
| `action_video_freq_ratio` | `4`; decode video offsets `0, 4, 8, 12, 16` |
| Action horizon | `16`; actions at offsets `0..15`, one command every 0.1 s |
| Action / state width | `8` / `8`; no OpenPI 32-D padding |
| Camera order | `left`, then `wrist`, concatenated horizontally |
| Final video size | `224 × 448` in height × width order |

Each camera is resized from 480 × 640 to 168 × 224, then padded by 28 pixels at the top and bottom to 224 × 224. The two cameras form one 224 × 448 RGB frame. This preserves the full field of view without cropping, rotating, or synthesizing a third camera. The video tensor has shape `[B, 3, 5, 224, 448]`; the action target has shape `[B, 16, 8]`. Five video frames satisfy Wan's `T % 4 == 1` requirement, and 16 actions are divisible by the four video transitions.

`RelativeJointTransform(keys=[default], joint_dim=7)` runs before normalization. For a chunk starting at frame `t`:

```python
relative_actions[:, :7] = absolute_actions[t : t + 16, :7] - state[t, :7]
relative_actions[:, 7] = absolute_actions[t : t + 16, 7]
```

Every target is relative to the same current joint state. This is not the difference between consecutive targets, and targets are not replaced with future measured states. State remains absolute. At episode boundaries the loader repeats terminal targets and supplies padding masks; `delta_action_dim_mask: null` retains those absolute targets before subtraction. Padded actions are excluded from action loss.

For a future controller integration, reverse normalization first and then add the raw current joint state to the first seven predicted dimensions. Preserve the eighth dimension. This is the representation contract, not an implemented robot execution client; the existing LIBERO/RoboTwin deployment interfaces are not FR3 execution entrypoints.

The reference package's active `deploy_fr3_chunk-final.py` constructs the final state dimension as `1` when `width > 0.05`, meaning open is 1. This is opposite to the recording convention. A future integration must reconcile that state construction before reusing the client. The client also sends each image already letterboxed to `224 × 224`; applying the raw `640 × 480` resize/pad transform again would add padding twice. These are observed differences between the existing client and the training data; this change does not modify the reference package or implement robot deployment.

The sources are under `dataset_real_world/fr3_example_deployment`: recording definitions in `gello_software/gello/agents/collect_fr3_gello_lerobot.py`, lines 11–15 and 253–269; client state construction in `irl_polymetis/polymetis/tests/deploy_fr3_chunk-final.py`, lines 134–138; and image preprocessing at lines 98–101. That client's action output at lines 317–323 still closes the gripper when `action[7] > 0.5`; the direction mismatch specifically concerns observation state.

## Statistics, model assets, and training entrypoint

The recipe uses global z-score normalization after the relative-joint transform, with EasyWAM's normalized-value clipping to `[-5, 5]`. With `data.train.pretrained_norm_stats: null`, dataset initialization computes EasyWAM statistics and saves `dataset_stats.json` under the training output directory. Statistics retain EasyWAM's per-episode aggregation and terminal-repetition semantics. Compute them from the full selected training corpus; a one-episode smoke file is insufficient. OpenPI statistics, LeRobot `episodes_stats.jsonl`, and LIBERO statistics are not substitutes. The reference OpenPI check used `skip_norm_stats=True` and did not validate this recipe's statistics.

[`scripts/run_real_world_fr3.sh`](../../../scripts/run_real_world_fr3.sh) is the new one-command entrypoint. It changes to the EasyWAM root, selects `.venv/bin/python`, and sets `PATH`/`PYTHONPATH`. It invokes [`run_real_world_fr3.py`](../../../scripts/run_real_world_fr3.py), which composes one final Hydra configuration and passes the same data, assets, cache settings, and user overrides to every stage. A failed subprocess prevents later stages. **With no stage argument, it defaults to `all` and starts training at the end.** Select `prepare` to stop after preparation. With the environment activated, `python scripts/run_real_world_fr3.py <stage>` is equivalent.

| Stage | Operation |
| --- | --- |
| `all` (default) | `convert → init-action → text-cache → train` |
| `prepare` | `convert → init-action → text-cache`, then exit |
| `config` | Resolve and print configuration without model construction or GPU access |
| `convert` | Convert every recording in the selected raw root to 10 Hz LeRobot v2.1 |
| `init-action` | Create a separate ActionDiT backbone initialization from the selected Robot video DiT |
| `text-cache` | Fill missing real masked embeddings using the selected local UMT5/tokenizer |
| `train` | Use the prepared dataset, ActionDiT, and cache with the native Accelerate/DeepSpeed ZeRO-1 trainer |

The older [`scripts/train_real_world_fr3.sh`](../../../scripts/train_real_world_fr3.sh) retains its `config`, `text-cache`, and `train` stages and displays help without arguments. Its defaults are the existing OpenPI reference dataset, `EasyWAM/checkpoints/Wan2.2-TI2V-5B`, and a pre-existing `ActionDiT_Wan22_5B_alphascale_1024hdim.pt`. It does not convert data or initialize ActionDiT. Commands and defaults below refer to the new workflow.

## Robot assets and separate ActionDiT initialization

On this machine, `FR3_CHECKPOINT_ROOT` defaults to `/path/to/storage/checkpoints`, or `EasyWAM/../../checkpoints`. The Robot video DiT is `checkpoints/Wan2.2-5B-Robot/checkpoint.safetensors`. That directory does not contain T5/VAE; these assets and the tokenizer default to `checkpoints/Wan2.2-TI2V-5B/` under the same checkpoint root.

| Environment variable | New workflow default or purpose |
| --- | --- |
| `EASYWAM_PYTHON` | Python used by the shell entrypoint, default `EasyWAM/.venv/bin/python` |
| `FR3_RAW_ROOT` | `EasyWAM/../../datasets/dataset_real_world/fr3_example` |
| `FR3_DATA_DIR` | `EasyWAM/data/real_world_fr3_10hz`, for conversion output and training input |
| `FR3_TEXT_CACHE` | `EasyWAM/data/text_embeds_cache/real_world_fr3/wan22` |
| `FR3_CHECKPOINT_ROOT` | The parent checkpoint root described above |
| `WAN22_MODEL_DIR` | `$FR3_CHECKPOINT_ROOT/checkpoints/Wan2.2-5B-Robot` |
| `WAN22_DIT_PATH` | `$WAN22_MODEL_DIR/checkpoint.safetensors` |
| `WAN22_T5_PATH` | `$FR3_CHECKPOINT_ROOT/checkpoints/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth` |
| `WAN22_VAE_PATH` | `$FR3_CHECKPOINT_ROOT/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` |
| `WAN22_TOKENIZER_DIR` | `$FR3_CHECKPOINT_ROOT/checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl` |
| `FR3_ACTION_DIT_PATH` | `EasyWAM/checkpoints/ActionDiT_FR3_Wan22_Robot_alphascale_1024hdim.pt` |
| `FR3_PREPROCESS_DEVICE` | ActionDiT initialization device, default `cpu`; this stage uses bfloat16 |
| `FR3_TEXT_DEVICE` / `FR3_TEXT_DTYPE` | Text computation defaults to `cpu` / `bfloat16`; this run used `cpu` / `float32` |
| `RUN_ID` / `FR3_OUTPUT_DIR` | Timestamp by default; output under `EasyWAM/runs/real_world_fr3_fastwam_idm_wan22/$RUN_ID` |
| `NPROC_PER_NODE` | Training processes per node; falls back to `PET_NPROC_PER_NODE`, then `1` |

The direct files map to `model.backbone.dit_path`, `text_encoder_path`, and `vae_path`; the tokenizer maps to `model.backbone.tokenizer_model_id`. The supplied Robot checkpoint does not need copying or renaming. Older configurations with the three file fields set to `null` retain standard filename lookup under `model_id`. The workflow does not download assets.

`init-action` invokes native `preprocess_action_dit_backbone.py --task real_world_fr3_fastwam_idm_wan22` with the same Hydra overrides. It interpolates the selected video DiT into the configured 1024-dimensional ActionDiT, enabling alpha scaling by default. The output is a separate shared-backbone initialization, leaving the Robot weights intact; it is not a trained FR3 checkpoint. An existing output file is reused directly. **Select a new `FR3_ACTION_DIT_PATH` when changing the source DiT or ActionDiT structure**: the workflow does not automatically verify an existing file's source.

## Run the workflow or reuse existing data

Inspect the configuration, then choose the full workflow or preparation only; the shell entrypoint selects the project environment automatically:

```bash
bash scripts/run_real_world_fr3.sh config

# Default all: convert, initialize, cache text, then start training.
FR3_DATA_DIR="$PWD/data/real_world_fr3_10hz" NPROC_PER_NODE=1 \
  bash scripts/run_real_world_fr3.sh

# Preparation only; exits before training.
FR3_DATA_DIR="$PWD/data/real_world_fr3_10hz" \
  bash scripts/run_real_world_fr3.sh prepare
```

`convert` requires exactly one output dataset. When it does not exist, every raw Parquet episode is converted. An existing output is reused only when `meta/info.json` and a complete `conversion_summary.json` exist and the summary matches the selected raw root, every source episode path, and 10 Hz output. A one-episode smoke conversion cannot stand in for the complete recording set. Choose a new `FR3_DATA_DIR` for other existing outputs; raw data and reference packages are not overwritten.

To use `OPENPI_REFERENCE` from above, skip conversion and prepare the initialization and text cache before training:

```bash
FR3_DATA_DIR="$OPENPI_REFERENCE" bash scripts/run_real_world_fr3.sh init-action
FR3_DATA_DIR="$OPENPI_REFERENCE" bash scripts/run_real_world_fr3.sh text-cache
FR3_DATA_DIR="$OPENPI_REFERENCE" NPROC_PER_NODE=1 \
  bash scripts/run_real_world_fr3.sh train
```

All stages accept trailing Hydra overrides; the last value of a key wins. Keep data, assets, caches, and structural overrides consistent when running stages separately. Set `RUN_ID` or `FR3_OUTPUT_DIR` to name the training output explicitly. For multi-node training from raw data, run `prepare` once in a single process against shared storage. With an existing OpenPI dataset, run `init-action` and `text-cache` once instead. After the shared dataset, ActionDiT, and caches are ready, run only `train` on each node; do not repeat `all`. Distributed settings continue through the native launcher's `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and corresponding `PET_*` variables. The workflow preserves `CUDA_VISIBLE_DEVICES`; select the allocated devices and matching process count for training.

Text caching defaults to `++overwrite=false`. It deduplicates prompts, shards them by rank, and filters existing files before loading assets. A rank with no missing entries skips T5/tokenizer resolution and loading while retaining the required collective statistics. The workflow still checks that the selected T5 file and tokenizer directory exist. Append `++overwrite=true` to regenerate entries. The standalone cache script still selects CPU/GPU automatically and defaults to bfloat16 when no device/dtype is specified; the new workflow explicitly defaults to CPU. To use CPU float32:

```bash
FR3_TEXT_DEVICE=cpu FR3_TEXT_DTYPE=float32 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  bash scripts/run_real_world_fr3.sh text-cache
```

The equivalent Hydra controls are `++text_encoder_device=cpu ++text_encoder_dtype=float32`. Compute dtype does not change the cache contract: contexts are stored in bfloat16 with boolean masks and zeroed padding. Training defaults to `load_text_encoder=false` and requires real embeddings beforehand. OpenPI tokens, historical padding-visible FastWAM caches, and synthetic test caches are not substitutes.

The FR3 task now directly selects `real_world_fr3` data and the `fastwam_idm_wan22` model, with explicit training parameters rather than LIBERO task inheritance. Defaults are batch size 16 per process, accumulation 1, learning rate `1e-4`, cosine scheduling, 20,000 optimizer updates, logging every 100 updates, checkpointing every 2,000, and evaluation every 1,000. These are initial settings without FR3 tuning or GPU-capacity validation. Both DiTs and the linear state projection train; VAE and text encoder remain frozen. Runtime sets `seed` before model construction so newly initialized action/state projections also use that seed.

Adjust per-process batch size and accumulation for the available memory, for example:

```bash
TRAIN_ARGS=(batch_size=2 gradient_accumulation_steps=8 num_workers=4)
bash scripts/run_real_world_fr3.sh config "${TRAIN_ARGS[@]}"
NPROC_PER_NODE=1 bash scripts/run_real_world_fr3.sh train "${TRAIN_ARGS[@]}"
```

The effective global batch is `batch_size × total training processes × gradient_accumulation_steps`, or 16 for this single-process example. To reduce activation memory, optionally set `vae_micro_batch_size=1` and `model.backbone.use_gradient_checkpointing=true`. These settings do not establish capacity or throughput for a particular GPU.

With `val_set_proportion: 0.0` and no separate `val` dataset, the trainer samples one training example per rank for generation, action-error, and VAE-reconstruction diagnostics. This is not held-out generalization, LIBERO simulation, or FR3 robot success evaluation. The default `eval_every=1000` retains these offline diagnostics; append `eval_every=0` to disable them. `eval_save_video=false` skips video files but still performs inference.

Outputs include `config.yaml`, newly computed `dataset_stats.json`, `train.log`, `checkpoints/weights/step_XXXXXX.pt`, and `checkpoints/state/step_XXXXXX/`. Multi-node logs use `train_node_<rank>.log`. The final `output_dir=...` override controls both training output and launcher logs. The default retention is the latest three weight/state checkpoints, configurable with `checkpoint_save_limit`.

## Resume training or initialize from weights

For a full resume, pass the `checkpoints/state/step_XXXXXX/` directory and the previous run's matching statistics. This example assumes the original run used batch size 2 and accumulation 8 above:

```bash
PREVIOUS_RUN=/absolute/path/to/previous-run
export FR3_DATA_DIR=/absolute/path/to/original-training-dataset
FR3_OUTPUT_DIR="$PWD/runs/fr3_idm_resume01" NPROC_PER_NODE=1 \
  bash scripts/run_real_world_fr3.sh train \
  "resume=$PREVIOUS_RUN/checkpoints/state/step_002000" \
  "data.train.pretrained_norm_stats=$PREVIOUS_RUN/dataset_stats.json" \
  batch_size=2 gradient_accumulation_steps=8 num_workers=4
```

The directory restores model, optimizer, scheduler, training step, and data-sampling progress. Keep task, data, statistics, seed, batch size, process count, accumulation, and training schedule consistent with the original run. `max_steps` is the total target, not additional updates. Sampling offsets depend on batch size and total process count; changing them does not preserve the same data trajectory.

For a fresh optimization run initialized from a compatible FR3 model, pass a weight file and its matching statistics:

```bash
FR3_OUTPUT_DIR="$PWD/runs/fr3_idm_from_weights01" NPROC_PER_NODE=1 \
  bash scripts/run_real_world_fr3.sh train \
  "resume=$PREVIOUS_RUN/checkpoints/weights/step_002000.pt" \
  "data.train.pretrained_norm_stats=$PREVIOUS_RUN/dataset_stats.json" \
  batch_size=2 gradient_accumulation_steps=8 num_workers=4
```

The file branch restores weights only, without optimizer, scheduler, or step. `model.skip_dit_load_from_pretrain=true` can avoid redundant DiT initialization only when a compatible complete model will subsequently be restored; VAE assets are still required.

An FR3 model with 8-D actions and states cannot strictly load a complete LIBERO checkpoint with 7-D actions and 8-D states because the action projection shapes differ. Shared Wan/ActionDiT backbones exclude those task-specific action projections and can initialize a new run. OpenPI model weights also belong to a different architecture. See the [FastWAM guide](../config/fastwam.md) for additional checkpoint semantics.

## Real text-cache execution and current validation scope

On 2026-10-11, the independent `.venv` loaded the real local UMT5 and tokenizer and computed embeddings on **CPU in float32 with eight threads**. The log reports `required=2 cached=0 to_encode=2`, followed by **`new=2 overwrite=0 skip=0`**, with no over-length prompts. Output under `EasyWAM/data/text_embeds_cache/real_world_fr3/wan22` uses training-compatible bfloat16 contexts, boolean masks, and zeroed padding. The [real text-cache log](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/precompute_real_text.log) preserves the asset paths, actual subprocess command, and results.

The two entries cover the real instruction `Insert the tubes into the rack one by one.` and the default-template prompt for the converter's empty hierarchical placeholder task. Task text came from metadata for the complete 100-episode OpenPI reference and the earlier real single-episode conversion at `/tmp/easywam-fr3-real-smoke-20261011`. These sources were used only to collect and deduplicate prompts. They do not form a training dataset containing duplicate episode 0, and this step did not reconvert all 100 episodes.

The [real cache consumer check](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/real_text_cache.json) passes: both contexts are bfloat16 `[128,4096]` with bool `[128]` masks. The real instruction has 28 valid tokens and the empty-placeholder template has 18; values are finite and padding is zero. The configured `RobotVideoDataset._get(0)` reads real episode 0 and returns video `[3,5,224,448]`, action/proprio `[16,8]`, and context/mask tensors that match the real disk cache element for element.

The [real cache reuse run](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/precompute_real_text_reuse.log) also exits with code 0: `required=2 cached=2 to_encode=0`, followed by `new=0 overwrite=0 skip=2`, without entering the `Preparing text encoder` branch. Skipping model loading therefore has both dedicated test coverage and a real cache rerun as evidence.

This round covers **45 passing CPU cases** across separate groups: [23 existing regression cases](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/pytest_existing_regression.log), nine asset/preprocessing cases, eight text-cache cases, and [five workflow cases](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/pytest_workflow.log).

All eight dedicated text-cache cases are covered by passing results: six passed initially; two distributed test cases failed because their test double omitted `get_rank()`, then passed when rerun after that repair. Both the [initial log](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/pytest_precompute_initial.log) and [retry log](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/pytest_precompute_retry.log) are preserved. The fully cached case forbids asset resolution, T5 loading, and tokenizer construction, verifying that model loading is skipped. Empty-rank collective statistics are also covered. The [nine CPU asset/preprocessing tests](../../../../test_artifacts/real_world_fr3_workflow_20261011_v1/asset_preprocess_cpu.json) check direct paths, task configuration, and native ActionDiT interpolation/alpha scaling with small test doubles; they do not execute full real ActionDiT initialization.

The completed work is workflow integration and real CPU text caching. Full reconversion, full real ActionDiT initialization, GPU model execution, GPU/multi-node training, throughput measurement, and robot execution remain **NOT-RUN**. The documented `all` command actually starts training when run; it does not indicate those stages have already executed.

## Historical CPU acceptance

The latest independent training environment acceptance is described above. The following preserves the earlier implementation checks in the temporary CPU environment: **16 tests passed** in 244.40 seconds, with one torchvision deprecation warning. Six new cases covered real configuration parsing in a subprocess, stubbed text-cache/training commands, output logs and GPU-environment preservation, and seeding before model construction. The seeding test mocks the model, Accelerator, and trainer; no real model is loaded. See the [training entrypoint acceptance record](../../../../test_artifacts/real_world_fr3_training_20261011_v1/RESULTS.json) and [pytest log](../../../../test_artifacts/real_world_fr3_training_20261011_v1/pytest.log). That run used the following command from the Outer root:

```bash
CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /tmp/easywam-fr3-cpu-20261011/bin/python -m pytest -q \
  EasyWAM/tests/test_real_world_fr3.py EasyWAM/tests/test_convert_real_world_fr3.py \
  EasyWAM/tests/test_fr3_openpi_parquet_metadata.py EasyWAM/tests/test_training_entrypoint.py \
  EasyWAM/tests/test_fr3_training_launcher.py
```

Both shell entrypoints pass `bash -n`. The converter, joint transform, and five test files—seven Python files in total—pass Ruff check. `runtime.py` retains 17 pre-existing lint findings with no added diagnostic code/message counts against `git HEAD`; the LeRobot reader retains the 11 findings recorded previously. Neither file with existing findings is claimed to pass whole-file lint.

The earlier data-integration evidence remains below; its reports and logs have not been overwritten.

The preceding conversion, transform, configuration, processor, and v2.1 metadata suite passed **10 tests** in 12.79 seconds, with one torchvision deprecation warning. Run the suite from the Outer repository root:

```bash
CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /tmp/easywam-fr3-cpu-20261011/bin/python -m pytest -q \
  EasyWAM/tests/test_real_world_fr3.py EasyWAM/tests/test_convert_real_world_fr3.py \
  EasyWAM/tests/test_fr3_openpi_parquet_metadata.py
```

The converter, joint transform, and three test files pass Ruff check. The native LeRobot reader retains its 11 pre-existing lint findings; comparing diagnostic code/message counts with `git HEAD` found no added findings. The temporary CPU environment uses the project's specified PyAV `16.0.1`; its initial inherited PyAV `12.3` lacked `codec.canonical_name`, so that environment mismatch was corrected before the passing rerun.

The [real episode 0 report](../../../../test_artifacts/real_world_fr3_20261011_v1/real_episode0.json) records a **PASS** for 1,469 raw frames converted to 490 frames. Every output state/action row exactly equals `float32(raw[::3])`; both videos contain 490 frames at 10 FPS and 640 × 480. Six first/middle/last-frame comparisons have RGB mean absolute error 2.024–2.280 on the 0–255 scale, consistent with lossy re-encoding. Direct reading of the existing OpenPI episode passes, and the newly converted episode passes the complete `RobotVideoDataset` path with statistics, processor, a synthetic text cache, `[3, 5, 224, 448]` video, `[16, 8]` actions, and terminal padding. The [validator](../../../../test_artifacts/real_world_fr3_20261011_v1/validate_real_episode0.py) and before/after-fix logs are retained with the report.

That real-data check covered one episode and used a synthetic text cache at the time. Real CPU UMT5 caching was completed subsequently, as recorded above; these historical checks do not validate full reconversion, GPU execution, training, or robot execution.
