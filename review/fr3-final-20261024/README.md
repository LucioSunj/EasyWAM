# FR3 Final FastWAM-IDM: Independent Review Snapshot

This branch publishes the actual EasyWAM source used by a running FR3 training
job, including pre-existing local research changes and the review fixes. It is
not a clean upstream checkout, a two-GPU adaptation, or a new training run.

- Repository: `LucioSunj/EasyWAM`
- Review branch: `codex/fr3-final-fastwam-idm-review-20261025`
- Base commit: `5aa63729250ecdf693cb6c553d7de8ad6eee30cf`
- Publication date: 2026-10-25 (Asia/Shanghai)
- Scope: read-only independent correctness review; do not launch another job.

## Read These First

1. `review/fr3-final-20261024/train_config.yaml`: the actual resolved four-GPU
   recipe. Publication replaces deployment paths with named environment
   variables. All non-path values and hyperparameters are intact.
2. `review/fr3-final-20261024/evidence.json`: bounded, completed test/probe
   evidence. Finite losses are not proof of convergence or absence of bugs.
3. `review/fr3-final-20261024/source_manifest.json`: SHA-256 identities of 174
   active source/config files checked against the launch acceptance manifest,
   plus the 47 working-tree changes over the base commit.
4. `src/model/fastwam.py`, `src/model/easywam_mot_idm.py`,
   `src/model/easywam_mot.py`, `src/model/component/mot.py`,
   `src/model/component/attention.py`, `src/model/component/action_dit.py`.
5. `src/runtime.py`, `src/trainer.py`, `src/utils/samplers.py`,
   `src/data/lerobot/robot_video_dataset.py`,
   `src/data/lerobot/processors/wam_processor.py`,
   `src/data/lerobot/transforms/relative_action.py`.
6. The tests named below and any dependencies needed to establish a claim.

The ordinary task template is not the source of truth for this run. Its batch
defaults differ. Use the frozen review configuration, with the launcher settings
below, when deciding whether a suspected issue is active.

The older FR3 workflow documentation contains dated environment and dataset
examples from October 11. Those historical CPU-only acceptance statements,
storage paths, and absent local report links are not the current run's evidence.
The numerical training/model implementation is preserved byte-for-byte. For
public release only, internal storage roots and raw dataset example names have
been generalized, and two physical camera identifiers in the conversion script
and examples are replaced by `000000000001` and `000000000002`. These are example
identifiers, not the real devices. Corresponding path tests were updated by the
same literal substitution. No server training files were changed. The manifest
records original and published hashes and every affected file. Use this review
directory for the final dataset and four-GPU acceptance facts.

## Exact Active Contract

| Property | Value |
| --- | --- |
| Model factory | `runtime.create_fastwam_idm` / `FastWAMIDM` |
| Trainer | Native EasyWAM `runtime.run_training` / `EasyWAMTrainer` |
| Hardware | Four NVIDIA A800 80GB GPUs |
| Microbatch / accumulation | 8 per GPU / 2; global batch 64 |
| Precision / parallelism | BF16 / DeepSpeed ZeRO-1 |
| Updates / LR / schedule | 20,000 / 1e-4 / cosine, warmup ratio 0.05 |
| Seed / weight decay / clipping | 42 / 0.01 / 1.0 |
| Resume | null; fresh Robot initialization, not a previous FR3 checkpoint |
| Trainable scope | Full video/action transformer parameters and state projection |
| Frozen components | VAE; cached text embeddings, no text-encoder training |
| Gradient checkpointing | Disabled for video and action backbones |
| Loss weights | action 1.0, video 1.0 |
| Conditional-video noise probability | 0.5 |
| Save / retention | Every 2,000 steps; three weight files, one complete state |
| Evaluation | Every 1,000 steps, ten inference steps, no saved evaluation video |

Runtime versions: Python 3.10.20, PyTorch 2.7.1+cu128, Accelerate 1.5.2,
DeepSpeed 0.16.9. The launcher uses
`scripts/accelerate_configs/accelerate_zero1_ds.yaml`, four processes, one
machine, and this frozen configuration. The environment retains
`NCCL_CUMEM_HOST_ENABLE=0`; no two-GPU allocator overrides were used.

## Dataset and Initialization Identity

The final LeRobot v2.1 dataset contains 101 episodes, 47,888 frames, and 202
videos at 10 Hz. All Parquet rows and all videos were checked, including full
video decoding. No dataset, image, video, or model weight is published here.

- Task: `Insert the tubes into the rack one by one.`
- Two cameras: left and wrist; source frames 640x480.
- Each camera is resized to 168x224 and padded to 224x224, then horizontally
  concatenated to 224x448.
- Seventeen data time points produce five video frames at offsets 0/4/8/12/16
  and sixteen 8-D actions.
- Actual sample shapes: video `[3,5,224,448]`, action `[16,8]`, proprio `[16,8]`,
  text context `[128,4096]`.
- Seven joint targets are relative to the chunk-start joint state. The gripper
  remains absolute. Normalization was recomputed from the final dataset.
- Archive SHA-256:
  `6eab4c5e720e634a0e6d19dfdfec88e0ea1ecb4ce64de9801fd6dd6c47e8a157`.

The video backbone is Wan2.2-5B-Robot, not the TI2V base DiT. The VAE/text
components are the compatible Wan2.2-TI2V-5B components. The 1024-hidden ActionDiT
backbone was derived from the selected Robot checkpoint: all 820 backbone tensors
were reconstructed with the repository's conversion algorithm and compared using
`torch.equal`, with zero mismatches. Fresh action/state input/output projections
are intentionally initialized from seed 42.

## Confirmed Issues Fixed Before Launch

### 1. Padding Could Influence Valid IDM Predictions

The previous implementation masked losses but left padded action/video tokens
visible as attention keys. A fixed-RNG test changed only padded values and
observed a change in valid losses.

`easywam_mot_idm.py` now canonicalizes padded video pixels before the temporal
VAE, maps padding to latent validity, and overlays per-sample key validity on the
teacher-forcing mask. `mot.py` accepts both `[S,S]` and `[B,S,S]` masks. The latter
uses the existing SDPA implementation rather than an incompatible FlashAttention
path. All-valid batches keep the structured-mask fast path.

Tests cover independent video branches, action-only/video-only gradient routes,
pad-value invariance, all-pad zero losses, and leading/internal padding. Four-GPU
full-size training deliberately exercised the padded path for three updates.

### 2. Odd Epoch Length Desynchronized Trainer and DeepSpeed Steps

The final dataset yields 1,497 global microbatches per epoch, which is odd with
accumulation two. With `sync_with_dataloader=True`, an actual four-rank probe
reproduced:

```text
OPTIMIZER_STEP_MISMATCH trainer_next=2 ds_updates=1 epoch=0 batch=3
```

The demonstrated defect is a counter mismatch, not evidence that DeepSpeed had
already performed a half-sized optimizer update. In this installed Accelerate
version the wrapper invokes `engine.backward` and `engine.step` for each
microbatch; DeepSpeed owns its accumulation boundary.

The frozen run sets `gradient_accumulation_sync_with_dataloader: false`, carrying
accumulation across epochs and preserving global batch 64. An actual four-rank
ZeRO-1 probe completed six updates, saved state, and resumed to eight with
matching trainer/engine counters on all ranks.

### 3. Checkpoint Completion and Legacy Counter Validation

All ranks finish `save_state()` and synchronize before rank zero publishes the
progress marker with temporary-file plus `os.replace`. Incomplete progress
metadata is rejected before loading state. Save/load validates the trainer and
DeepSpeed global-step agreement. New complete state is established before old
state is pruned; state and ordinary weights have independent retention limits.

### 4. FR3 Postprocessing Returned an Empty Action Horizon

The WAM processor sliced sixteen predicted actions starting at
`num_obs_steps - 1 == 16`. It now retains the full current-conditioned future
chunk. Tests cover shape and relative/normalization inversion. This was a latent
deployment issue; the active training loss did not call this postprocess path.

### 5. Proprio Padding Mask Had the Wrong Length

`RobotVideoDataset` trimmed proprio from seventeen to sixteen steps without
trimming `proprio_is_pad`. Both are now trimmed together and covered by a test.

## Evidence and Deliberate Limits

- The full source test suite passed: 208 tests, six dependency/CPU warnings.
- The metadata-sanitized publication copy also passed all 208 tests in an
  isolated CPU-only checkout; see `pytest-publication.txt`. An initial incomplete
  test fixture omitted the unchanged `experiments/` package; restoring it from
  the base commit fixed those three import failures without changing code.
- Real complete-model four-GPU B8/accumulation2 padding stress completed three
  updates with finite losses/gradients and nonzero state/action-head gradients.
- A separate complete-model four-GPU probe performed one update followed by the
  unchanged ten-step inference/evaluation path. CUDA all-false SDPA rows produced
  finite zero outputs and gradients.
- Real four-rank ZeRO-1 small-model save/resume to step eight passed. This does
  not substitute for the first complete large-model checkpoint at step 2,000.
- Formal training was freshly initialized after the probes; probe monkeypatches,
  forced padding and shortened step counts were not used by the formal job.

Known limitations, not assertions of a clean bill of health:

1. `val_set_proportion=0.0`; evaluation samples are training data, not independent
   validation. Do not infer generalization or robot success from these metrics.
2. Normalization remains the original episode-weighted EasyWAM convention.
3. Read/processing failures are logged and may randomly substitute samples; the
   validated corpus/probes did not trigger these failures.
4. Some non-unit inference-guidance arguments are unsupported. Active evaluation
   uses guidance scales of one.
5. Evaluation action metrics are not a complete terminal-padding-aware benchmark.
6. Bitwise resumption under persistent workers/prefetch, changed data/seed/schedule,
   or arbitrary legacy metadata has not been established.
7. Full-size optimizer-state checkpoint I/O had not been exercised at publication.

## Independent Reviewer Request

Prioritize concrete active bugs over generic hardening suggestions. For each
finding give the exact file/function/lines, information or state-flow argument,
triggering condition, whether the frozen four-GPU configuration triggers it, and
a minimal falsifiable regression test. Distinguish confirmed bugs, hypotheses,
and inactive limitations. Inspect dependencies before making claims.

In particular challenge padding through the VAE and multi-layer attention,
teacher-forcing gradient routes, state conditioning, flow-matching targets and
weights, and the exact Accelerate/DeepSpeed save/resume protocol. State whether a
finding warrants stopping the currently running job, can wait for a future
restart, or only calls for more evidence. Do not recommend throwing away a run
solely because a theoretical risk exists. Do not edit or launch anything.
