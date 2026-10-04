"""Small real transformer models; no pretrained assets or GPU are used."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from model.backbone.wan22.wan_video_dit import WanVideoDiT
from model.component.mot import MoT

VIDEO_CONFIG = {
    "hidden_dim": 16,
    "in_dim": 3,
    "out_dim": 3,
    "ffn_dim": 32,
    "text_dim": 12,
    "freq_dim": 8,
    "eps": 1e-6,
    "patch_size": (1, 2, 2),
    "num_heads": 2,
    "attn_head_dim": 8,
    "num_layers": 2,
    "has_image_input": False,
    "seperated_timestep": True,
    "video_attention_mask_mode": "first_frame_causal",
}
ACTION_CONFIG = {
    "hidden_dim": 8,
    "action_dim": 3,
    "ffn_dim": 16,
    "text_dim": 12,
    "freq_dim": 8,
    "eps": 1e-6,
    "num_heads": 2,
    "attn_head_dim": 8,
    "num_layers": 2,
}


class TinyVAE(nn.Module):
    temporal_downsample_factor = 4
    spatial_downsample_factor = 8
    upsampling_factor = 8
    z_dim = 3

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.model = SimpleNamespace(z_dim=3)

    def encode(self, videos, device=None, **kwargs):
        if isinstance(videos, list):
            videos = torch.stack(videos)
        return F.avg_pool3d(videos[:, :, ::4], (1, 8, 8)) * self.scale

    def decode(self, latents, device=None, **kwargs):
        return F.interpolate(
            latents,
            size=((latents.shape[2] - 1) * 4 + 1, 16, 32),
            mode="trilinear",
            align_corners=False,
        )


def tokenize(prompts, **kwargs):
    if isinstance(prompts, str):
        prompts = [prompts]
    ids = torch.tensor([[1, 2, 3 if p == "long" else 0, 0] for p in prompts])
    return ids, ids != 0


class TinyTextEncoder(nn.Embedding):
    def __init__(self):
        super().__init__(4, 12)

    def forward(self, ids, mask):
        return super().forward(ids)


def make_model(model_cls, *, checkpointing=False):
    torch.manual_seed(13)
    video = WanVideoDiT(**VIDEO_CONFIG, use_gradient_checkpointing=checkpointing)
    action = model_cls.action_expert_cls(
        **ACTION_CONFIG, use_gradient_checkpointing=checkpointing
    )
    return model_cls(
        video_expert=video,
        action_expert=action,
        mot=MoT({"video": video, "action": action}),
        vae=TinyVAE(),
        text_encoder=TinyTextEncoder(),
        tokenizer=tokenize,
        state_dim=4,
        text_dim=12,
        device="cpu",
        torch_dtype=torch.float32,
    )


def make_sample():
    generator = torch.Generator().manual_seed(27)
    return {
        "video": torch.randn(2, 3, 5, 16, 32, generator=generator),
        "action": torch.randn(2, 4, 3, generator=generator),
        "proprio": torch.randn(2, 5, 4, generator=generator),
        "context": torch.randn(2, 4, 12, generator=generator),
        "context_mask": torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        ),
        "action_is_pad": torch.tensor([[False] * 4, [False, False, False, True]]),
        "image_is_pad": torch.zeros(2, 5, dtype=torch.bool),
    }
