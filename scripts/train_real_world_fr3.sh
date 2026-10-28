#!/usr/bin/env bash
# FR3 training workflow using EasyWAM's native Hydra, text cache and trainer.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/train_real_world_fr3.sh {config|text-cache|train} [Hydra overrides...]

  config      Print the resolved training config; do not construct a model.
  text-cache  Generate masked UMT5 embeddings with the configured Wan weights.
  train       Launch the native Accelerate/DeepSpeed ZeRO-1 trainer.

Paths (environment variables):
  FR3_DATA_DIR        Existing 10 Hz LeRobot root. Defaults to the supplied
                      OpenPI conversion reference under datasets/dataset_real_world.
  FR3_TEXT_CACHE     Text cache directory (default: data/text_embeds_cache/real_world_fr3/wan22).
  WAN22_MODEL_DIR    Wan2.2-TI2V-5B directory (default: checkpoints/Wan2.2-TI2V-5B).
  FR3_ACTION_DIT_PATH  Preprocessed ActionDiT backbone checkpoint.
  FR3_OUTPUT_DIR    Training output directory (default: runs/<FR3 task>/<RUN_ID>).

Activate the EasyWAM Python environment before running this script.
Set CUDA_VISIBLE_DEVICES and NPROC_PER_NODE for your allocated training devices.
NPROC_PER_NODE defaults to 1; existing distributed launcher variables are preserved.
All three stages accept the same Hydra data/model overrides.
EOF
}

STAGE="${1:-help}"
case "${STAGE}" in
  -h|--help|help) usage; exit 0 ;;
  config|text-cache|train) shift ;;
  *) usage >&2; exit 2 ;;
esac

EASYWAM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${EASYWAM_ROOT}"
export PYTHONPATH="${EASYWAM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

TASK=real_world_fr3_fastwam_idm_wan22
REFERENCE_NAME=fr3_example
REFERENCE_ROOT="${EASYWAM_ROOT}/../../datasets/dataset_real_world/${REFERENCE_NAME}_conversion_reference_20261011"
FR3_DATA_DIR="${FR3_DATA_DIR:-${REFERENCE_ROOT}/converted_dataset/lsk/${REFERENCE_NAME}}"
FR3_TEXT_CACHE="${FR3_TEXT_CACHE:-${EASYWAM_ROOT}/data/text_embeds_cache/real_world_fr3/wan22}"
WAN22_MODEL_DIR="${WAN22_MODEL_DIR:-${EASYWAM_ROOT}/checkpoints/Wan2.2-TI2V-5B}"
FR3_ACTION_DIT_PATH="${FR3_ACTION_DIT_PATH:-${EASYWAM_ROOT}/checkpoints/ActionDiT_Wan22_5B_alphascale_1024hdim.pt}"
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
FR3_OUTPUT_DIR="${FR3_OUTPUT_DIR:-${EASYWAM_ROOT}/runs/${TASK}/${RUN_ID}}"

COMMON_ARGS=(
  "task=${TASK}"
  "data.train.dataset_dirs=[${FR3_DATA_DIR}]"
  "data.train.text_embedding_cache_dir=${FR3_TEXT_CACHE}"
  "model.backbone.model_id=${WAN22_MODEL_DIR}"
  "model.backbone.tokenizer_model_id=${WAN22_MODEL_DIR}/google/umt5-xxl"
  "model.backbone.action_dit_pretrained_path=${FR3_ACTION_DIT_PATH}"
  "output_dir=${FR3_OUTPUT_DIR}"
)

case "${STAGE}" in
  config)
    exec python scripts/train.py --cfg job --resolve "${COMMON_ARGS[@]}" "$@"
    ;;
  text-cache)
    exec python scripts/precompute_text_embeds.py "${COMMON_ARGS[@]}" ++overwrite=false "$@"
    ;;
  train)
    export NPROC_PER_NODE="${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-1}}"
    exec bash scripts/train_zero1.sh "${COMMON_ARGS[@]}" "$@"
    ;;
esac
