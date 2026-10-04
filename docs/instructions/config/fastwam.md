# FastWAM models on EasyWAM

[中文](fastwam_zh.md) | [Model configuration](models.md) | [Configuration index](README.md)

The Wan2.2 FastWAM, FastWAM-Joint and FastWAM-IDM models retain the original
linear action encoder, action output head and proprioception projection. They
use EasyWAM's trainer, data pipeline, structured attention, VAE batching,
cross-attention K/V reuse, compilation targets and dynamic inference batching.
The original FastWAM Python package is not a runtime dependency.

**Text padding follows EasyWAM:** only valid text tokens participate in
attention, for both cached training contexts and online prompt encoding. This
deliberately changes the padding-visible behavior of original FastWAM
checkpoints. Loading an old checkpoint imports its weights, not its historical
predictions or evaluation scores.

## Recipes

| Model | LIBERO task | RoboTwin task |
| --- | --- | --- |
| FastWAM / UNCOND | `libero_fastwam_wan22` | `robotwin_fastwam_wan22` |
| FastWAM-Joint | `libero_fastwam_joint_wan22` | `robotwin_fastwam_joint_wan22` |
| FastWAM-IDM | `libero_fastwam_idm_wan22` | `robotwin_fastwam_idm_wan22` |

These tasks inherit the corresponding EasyWAM data and training recipes,
including their step budgets; they do not resume an earlier experiment's
scientific settings. The Video DiT, Action DiT and linear state projection are
trainable; the VAE and text encoder remain frozen. State tokens stay in the
text context (`state_position: context`). The IDM recipe retains conditional
video noise probability `0.5`.

Inspect the composed recipe without loading a model or starting training:

```bash
python scripts/train.py --cfg job task=libero_fastwam_idm_wan22
```

Follow the [Wan2.2 preparation guide](../backbone/wan22.md) and the appropriate
[data guide](../data/README.md). The interpolated ActionDiT backbone contains
the shared transformer weights and works with the linear FastWAM projections.
The preparation script also resolves the new inherited model configs:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam_wan22.yaml \
  --backbone wan22 \
  --output checkpoints/ActionDiT_Wan22_5B_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

Use EasyWAM's text-cache generation command for the selected data and model:

```bash
python scripts/precompute_text_embeds.py task=libero_fastwam_idm_wan22
```

Old FastWAM cache filenames and padding-visible masks are not the new cache
contract. Recompute caches with EasyWAM instead of renaming old files. To use
an existing FastWAM dataset, override `data.train.dataset_dirs`; when initializing
from trained weights, also supply the checkpoint's normalization statistics
through `data.train.pretrained_norm_stats`.

Launch with the usual EasyWAM distributed scripts after selecting resources:

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=libero_fastwam_idm_wan22
```

See [efficiency configuration](efficiency.md) for attention backend, VAE batch
size, gradient checkpointing and data-loader controls. The migrated models use
the same controls; their speed on a particular workload must be measured.

## Initialize from an original checkpoint

Original full-model checkpoints contain `mot` and, when enabled,
`proprio_encoder`. The FastWAM loader imports all MoT tensors strictly and maps
the linear proprioception weights to EasyWAM's `state_encoder`. Missing model
weights fail loading instead of leaving randomly initialized layers.

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_fastwam_idm_wan22 \
  resume=/path/to/original_fastwam_idm.pt \
  model.skip_dit_load_from_pretrain=true \
  +data.train.pretrained_norm_stats=/path/to/dataset_stats.json
```

Select the matching UNCOND, Joint or IDM recipe: old checkpoints do not identify
their variant. A `.pt` `resume` initializes weights with fresh training state.
Use the native EasyWAM `checkpoints/state/step_*` directory to resume optimizer,
scheduler, RNG and data-loader position from a run started with these recipes.
New weight checkpoints record the model variant, `state_position: context`
and `text_padding: masked`.

## Inference

The standard evaluation entry points accept the new task names:

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_fastwam_idm_wan22 \
  ckpt=/path/to/checkpoint.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json
```

`infer_action_batch` and `infer_joint_batch` accept per-sample seeds and run
through the native batching/caching paths. Compilation uses the inherited
architecture-specific targets. Standard LIBERO, LIBERO-Plus and RoboTwin
evaluation configuration applies. Batched throughput and single-environment
latency are different measurements.

The provided tasks cover the original three supervised model variants.
Project-specific Action-only/dual LoRA BC, causal experiments and Gate/Flow-PPO
training require their own loss and parameter-ownership integration; the native
EasyWAM Video-LoRA recipe is not equivalent to those methods.

CPU tests use tiny real transformer blocks and synthetic VAE/text components:

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_fastwam.py
```

They verify strict checkpoint import, masked padding, training and gradients,
batched inference, recipe composition and native training-state restoration.
They do not establish production GPU performance or benchmark success rates.
