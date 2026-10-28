# FR3 Insert Tubes H32 Training Snapshot

This is the source and resolved recipe used by the Insert IDM run started on
2026-10-28. It is based on the previously published FR3 review and includes the
subsequent checkpoint, padding-aware evaluation, normalization, and raw-anchor
fixes. It does not contain datasets, model weights, credentials, SSH settings,
runtime authorization records, or checkpoint payloads.

## Recipe

- Four GPUs, batch 4 per GPU, gradient accumulation 4, global batch 64.
- BF16, DeepSpeed ZeRO-1, activation checkpointing disabled.
- 33 observation frames, nine video frames, 32 action steps, 10 Hz.
- Unclipped z-score continuous action/state normalization, fixed binary gripper
  mapping, and a raw chunk-start joint-state anchor.
- Fresh Wan2.2-5B-Robot and ActionDiT initialization, `resume: null`, 20,000 steps.
- One rolling full state and three weights; successful save precedes pruning.
- `NCCL_CUMEM_HOST_ENABLE=0`; P2P is not disabled.
- The sealed dataset has 299 episodes and 132,292 frames. H32 statistics SHA-256:
  `ae6728618868cbf8aa151825b75c8cee48fb39c33050efd43fecdb482684d7ae`.

`train_config.json` preserves every runtime setting except ten external path
values, represented as OmegaConf environment interpolations. The model,
dataset, cache, output, and statistics paths must be supplied explicitly. Do not
substitute H16 statistics, Drawer weights, or preflight weights for this recipe.

## Source Identity

`source_manifest.json` binds all 309 sealed source files and 15 control files to
their runtime and publication SHA-256 values. Public-copy redactions are limited
to internal storage roots, physical camera identifiers, and private example
names. The active remote source and process state are not modified by publishing.

The `controls/` directory is an archival implementation reference, not a new
deployment bundle. Its quota, one-shot, authorization, and integrity gates remain
fail-closed. The private sealed input and handoff records are intentionally not
published; historical hashes are not valid authorization for another run.
Do not execute these archived queue or stage wrappers against a live deployment.

## Evidence Boundary

Before the formal run, 28 remote CPU control tests passed, full-size fresh
updates 0 to 3 completed with exit 0, and an independent full-state resume from
3 to 4 completed with exit 0. Four-rank recovery and checkpoint audits passed.
Formal training then started from step zero, independently of those preflights.
This snapshot does not claim that the formal 20,000-step run is complete.

The preceding Drawer run was deliberately stopped at step 11,158, retaining only
its complete step-10,000 state. This was an authorized early switch, not Drawer
20,000-step completion. No Drawer state is used to initialize formal Insert.

## Integration

This runtime snapshot is integrated with `codex/fr3-insert-uncond-bc-20261028`
without deploying the merge back into the running job. The merge retains that
branch's absolute-gripper relative-joint support and optional unclipped
normalization. The runtime manifest refers to this snapshot before integration;
any integration-only changes are documented and tested separately.
