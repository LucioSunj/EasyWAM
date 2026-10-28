import argparse
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from model.backbone.loader import load_easywam_backbone
from model.component.action_dit import ActionDiT
from utils.config_resolvers import register_default_resolvers


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(
        f"Unsupported dtype: {name}. Expected one of: float32, float16, bfloat16."
    )


def _parse_bool(name: str) -> bool:
    value = str(name).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {name}")


def _is_unresolved_interpolation(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _resolve_from_video_cfg(value: Any, video_cfg: dict[str, Any]) -> Any:
    if not _is_unresolved_interpolation(value):
        return value
    text = str(value).strip()
    if not (text.startswith("${") and text.endswith("}")):
        return value
    expr = text[2:-1]
    if not expr.startswith("video_dit_config."):
        return value
    key = expr.split(".", 1)[1]
    if key not in video_cfg:
        return value
    resolved = video_cfg[key]
    return value if _is_unresolved_interpolation(resolved) else resolved


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(
    src: torch.Tensor, target_shape: tuple[int, ...]
) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        # Interpolate one dimension at a time and restore the original axis order.
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    return out.to(dtype=src.dtype)


def _load_model_config(
    path: Path | None,
    backbone_name: str | None,
    overrides: list[str] | None = None,
    *,
    task: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    register_default_resolvers()
    if task is not None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        config_name = "train"
        initial_overrides = [f"task={task}"]
        if backbone_name is not None:
            initial_overrides.append(f"model/backbone={backbone_name}")
    else:
        if path is None:
            raise ValueError("Specify a training task or model config path.")
        config_dir = path.resolve().parent.parent
        config_name = None
        initial_overrides = [
            f"+model={path.stem}",
            f"model/backbone={backbone_name or 'wan22'}",
        ]
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        root = compose(
            config_name=config_name, overrides=initial_overrides + list(overrides or [])
        )
    cfg = root.model
    if "action_dit_config" not in cfg:
        raise ValueError(f"`{task or path}` must contain `action_dit_config`.")
    action_cfg = OmegaConf.to_container(cfg.action_dit_config, resolve=False)
    if task is None and _is_unresolved_interpolation(action_cfg.get("action_dim")):
        print(
            "[WARN] `action_dit_config.action_dim` is unresolved; defaulting to 7 for preprocessing."
        )
        cfg.action_dit_config.action_dim = 7
    backbone_cfg = OmegaConf.to_container(cfg.backbone, resolve=True)
    action_cfg = OmegaConf.to_container(cfg.action_dit_config, resolve=True)
    return backbone_cfg, action_cfg


def _save_payload_atomic(payload: dict[str, Any], output_path: Path) -> None:
    """Publish only a complete payload so interrupted writes are not reused."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, str(temporary_path))
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _cosmos_source_key(target_key: str) -> str | None:
    parts = target_key.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        return None
    layer = parts[1]
    suffix = ".".join(parts[2:])
    mapping = {
        "self_attn.q.weight": "self_attn.q_proj.weight",
        "self_attn.k.weight": "self_attn.k_proj.weight",
        "self_attn.v.weight": "self_attn.v_proj.weight",
        "self_attn.o.weight": "self_attn.output_proj.weight",
        "self_attn.norm_q.weight": "self_attn.q_norm.weight",
        "self_attn.norm_k.weight": "self_attn.k_norm.weight",
        "cross_attn.q.weight": "cross_attn.q_proj.weight",
        "cross_attn.k.weight": "cross_attn.k_proj.weight",
        "cross_attn.v.weight": "cross_attn.v_proj.weight",
        "cross_attn.o.weight": "cross_attn.output_proj.weight",
        "cross_attn.norm_q.weight": "cross_attn.q_norm.weight",
        "cross_attn.norm_k.weight": "cross_attn.k_norm.weight",
        "ffn.0.weight": "mlp.layer1.weight",
        "ffn.2.weight": "mlp.layer2.weight",
    }
    source_suffix = mapping.get(suffix)
    return None if source_suffix is None else f"blocks.{layer}.{source_suffix}"


def _convert_backbone_state(
    backbone_name: str,
    video_state: dict[str, torch.Tensor],
    action_state: dict[str, torch.Tensor],
    backbone_keys: set[str],
    apply_alpha_scaling: bool,
) -> tuple[dict[str, torch.Tensor], int, int, int]:
    output = {}
    copied = interpolated = initialized = 0
    for key in sorted(backbone_keys):
        target = action_state[key]
        source_key = key if backbone_name == "wan22" else _cosmos_source_key(key)
        src = video_state.get(source_key) if source_key is not None else None
        if src is None:
            if backbone_name == "cosmos25":
                value = target
            elif (
                key == "text_embedding.2.weight"
                and target.ndim == 2
                and target.shape[0] == target.shape[1]
            ):
                value = torch.eye(target.shape[0], dtype=target.dtype)
            elif key.endswith(".weight") and ("norm3" in key):
                value = torch.ones_like(target)
            else:
                value = torch.zeros_like(target)
            initialized += 1
        elif (
            backbone_name == "cosmos25"
            and (".norm_q.weight" in key or ".norm_k.weight" in key)
            and src.ndim == target.ndim == 1
            and target.numel() % src.numel() == 0
        ):
            value = src.repeat(target.numel() // src.numel())
            interpolated += 1
        elif tuple(src.shape) == tuple(target.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(target.shape))
            if (
                apply_alpha_scaling
                and src.ndim >= 2
                and src.shape[-1] != target.shape[-1]
            ):
                value = (
                    value.float()
                    * (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                )
            interpolated += 1
        output[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()
    return output, copied, interpolated, initialized


def _require_int_config(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return int(value)


def _require_float_config(cfg: dict[str, Any], key: str) -> float:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ActionDiT weights from a Wan22 or Cosmos25 video backbone."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model-config",
        help="Path to model yaml, e.g. configs/model/easywam_mot_wan22.yaml",
    )
    source.add_argument(
        "--task", help="Compose configs/train.yaml with this training task."
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override; repeat for each argument.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .pt path for preprocessed ActionDiT backbone.",
    )
    parser.add_argument(
        "--backbone",
        choices=["wan22", "cosmos25"],
        help="Backbone override; defaults to the task's backbone, or wan22 with --model-config.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device for loading model and preprocessing."
    )
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Whether to apply alpha=sqrt(dv/da) when the last dimension is resized (true/false). Default: true.",
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    model_config_path = Path(args.model_config) if args.model_config else None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)

    backbone_cfg, action_cfg = _load_model_config(
        model_config_path, args.backbone, args.override, task=args.task
    )
    backbone_name = backbone_cfg["name"]
    if backbone_name not in ("wan22", "cosmos25"):
        raise ValueError(
            f"ActionDiT preprocessing does not support backbone {backbone_name!r}."
        )
    torch_dtype = _parse_dtype(args.dtype)
    int_fields = [
        "hidden_dim",
        "action_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
    ]
    for key in int_fields:
        action_cfg[key] = _require_int_config(action_cfg, key)
    action_cfg["eps"] = _require_float_config(action_cfg, "eps")

    print(
        f"[INFO] Loaded model config from {args.task or model_config_path}. "
        f"Preprocessing ActionDiT backbone with dtype={torch_dtype} on device={args.device}, "
        f"apply_alpha_scaling={apply_alpha_scaling}."
    )
    components = load_easywam_backbone(
        backbone_cfg,
        device=args.device,
        torch_dtype=torch_dtype,
    )
    video_expert = components.dit

    action_expert = ActionDiT(**action_cfg).to(device=args.device, dtype=torch_dtype)
    if int(action_cfg["num_heads"]) != int(video_expert.num_heads):
        raise ValueError(
            "ActionDiT `num_heads` must match video expert for MoT mixed attention."
        )
    if int(action_cfg["attn_head_dim"]) != int(video_expert.attn_head_dim):
        raise ValueError(
            "ActionDiT `attn_head_dim` must match video expert for MoT mixed attention."
        )
    if int(action_cfg["num_layers"]) != len(video_expert.blocks):
        raise ValueError("ActionDiT `num_layers` must match video expert.")

    action_state = action_expert.state_dict()
    video_state = video_expert.state_dict()
    backbone_keys = ActionDiT.backbone_key_set(action_state.keys())

    backbone_state_dict, copied, interpolated, initialized = _convert_backbone_state(
        backbone_name, video_state, action_state, backbone_keys, apply_alpha_scaling
    )

    payload = {
        "policy": {
            "skip_prefixes": list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
            "source_backbone": backbone_name,
            "mapping_version": 2 if backbone_name == "cosmos25" else 1,
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        },
    }
    _save_payload_atomic(payload, output_path)

    skipped = len(action_state) - len(backbone_keys)
    print(
        "[INFO] Saved ActionDiT backbone payload to "
        f"{output_path} (copied={copied}, interpolated={interpolated}, "
        f"initialized={initialized}, skipped={skipped})."
    )


if __name__ == "__main__":
    main()
