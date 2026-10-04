"""FastWAM architectures using EasyWAM training, attention and inference.

The original linear action/state projections are retained. Text padding uses
EasyWAM's valid-token masks, including when initializing from FastWAM weights.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from utils.logging_config import get_logger

from .component.action_dit import ActionDiT, validate_checkpoint_state_position
from .easywam_mot import EasyWAMMoT
from .easywam_mot_idm import EasyWAMMoTIDM
from .easywam_mot_joint import EasyWAMMoTJoint

logger = get_logger(__name__)


class FastWAMActionDiT(ActionDiT):
    """Original FastWAM projections with EasyWAM's attention and K/V cache."""

    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        del self.action_decoder
        self.head = nn.Linear(self.hidden_dim, self.action_dim)

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)


class _FastWAMArchitecture:
    """Shared FastWAM projections and strict full-model checkpoint loading."""

    action_expert_cls = FastWAMActionDiT

    def __init__(self, *args: Any, state_position: str = "context", **kwargs: Any):
        if state_position != "context":
            raise ValueError(
                "FastWAM keeps its state token in context; use state_position=context."
            )
        super().__init__(*args, state_position=state_position, **kwargs)
        if self.backbone_name != "wan22":
            raise ValueError("FastWAM models require the Wan2.2 backbone.")

    @classmethod
    def from_backbone_pretrained(cls, backbone: dict[str, Any], **kwargs: Any):
        if backbone["name"] != "wan22":
            raise ValueError("FastWAM models require the Wan2.2 backbone.")
        return super().from_backbone_pretrained(backbone, **kwargs)

    def _build_state_encoder(self, output_dim: int) -> nn.Module:
        return nn.Linear(self.state_dim, output_dim)

    def save_checkpoint(self, path, optimizer=None, step=None):
        from .component.lora import build_lora_checkpoint_payload, has_lora

        if has_lora(self.video_dit):
            payload = build_lora_checkpoint_payload(self, step=step)
        else:
            payload = {
                "mot": self.mot.state_dict(),
                "step": step,
                "torch_dtype": str(self.torch_dtype),
            }
            if self.state_encoder is not None:
                payload["state_encoder"] = self.state_encoder.state_dict()
        payload.update(
            backbone_name="wan22",
            model_variant=self.model_variant,
            state_position="context",
            text_padding="masked",
        )
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, merge_lora: bool = False):
        from .component.lora import (
            load_lora_model_checkpoint_state,
            load_standard_state_dict,
        )

        payload = torch.load(path, map_location="cpu", mmap=True)
        validate_checkpoint_state_position(payload, self.state_position)
        for key, expected in (
            ("backbone_name", "wan22"),
            ("model_variant", self.model_variant),
            ("text_padding", "masked"),
        ):
            if key in payload and payload[key] != expected:
                raise ValueError(
                    f"Checkpoint {key}={payload[key]!r}; expected {expected!r}."
                )

        if "proprio_encoder" in payload:
            logger.info(
                "Initializing FastWAM weights with EasyWAM text-padding masks. "
                "This changes the original padding-visible model behavior."
            )

        if payload.get("format") == "trainable_lora":
            load_lora_model_checkpoint_state(
                self, payload, merge_after_load=bool(merge_lora)
            )
        else:
            if "mot" not in payload:
                raise ValueError(
                    "FastWAM initialization requires a complete `mot` checkpoint."
                )
            state = payload.get("state_encoder", payload.get("proprio_encoder"))
            if (self.state_encoder is None) != (state is None):
                raise ValueError(
                    "Checkpoint and model must both include or both omit state weights."
                )
            load_standard_state_dict(self.mot, payload["mot"], strict=True)
            if self.state_encoder is not None:
                self.state_encoder.load_state_dict(state, strict=True)

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload


class FastWAM(_FastWAMArchitecture, EasyWAMMoT):
    """FastWAM with current-frame-conditioned actions and masked text padding."""

    model_variant = "fastwam"


class FastWAMJoint(_FastWAMArchitecture, EasyWAMMoTJoint):
    """FastWAM with joint video/action denoising and masked text padding."""

    model_variant = "fastwam_joint"


class FastWAMIDM(_FastWAMArchitecture, EasyWAMMoTIDM):
    """FastWAM teacher-forced IDM with EasyWAM batched video/action inference."""

    model_variant = "fastwam_idm"
