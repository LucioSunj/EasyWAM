from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pil_frames_to_video_tensor(frames: Sequence[Image.Image]) -> torch.Tensor:
    if len(frames) == 0:
        raise ValueError("`frames` must be non-empty.")

    frame_tensors = []
    for frame in frames:
        arr = np.array(frame.convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3, H, W]
        frame_tensors.append(x)
    return torch.stack(frame_tensors, dim=1)  # [3, T, H, W]


def _gaussian_kernel_2d(kernel_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    return kernel_2d.repeat(channels, 1, 1, 1)


def _valid_frame_mask(
    valid_frames: Optional[torch.Tensor],
    num_frames: int,
    *,
    exclude_first_frame: bool,
) -> torch.Tensor:
    """Normalize a frame-validity mask without allowing implicit broadcasting."""
    if valid_frames is None:
        mask = torch.ones(num_frames, dtype=torch.bool)
    else:
        if not isinstance(valid_frames, torch.Tensor):
            raise TypeError(
                "`valid_frames` must be a torch.Tensor or None, "
                f"got {type(valid_frames)}."
            )
        if valid_frames.ndim == 2:
            if valid_frames.shape[0] != 1:
                raise ValueError(
                    "`valid_frames` may be [T] or a single batched [1, T] mask, "
                    f"got shape {tuple(valid_frames.shape)}."
                )
            valid_frames = valid_frames[0]
        if valid_frames.ndim != 1 or valid_frames.shape[0] != num_frames:
            raise ValueError(
                "`valid_frames` must have shape [T] (or [1, T]), "
                f"got {tuple(valid_frames.shape)} for T={num_frames}."
            )
        mask = valid_frames.detach().to(device="cpu", dtype=torch.bool)

    if exclude_first_frame and num_frames > 0:
        mask = mask.clone()
        mask[0] = False
    return mask


def video_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
    *,
    valid_frames: Optional[torch.Tensor] = None,
    exclude_first_frame: bool = False,
    return_count: bool = False,
) -> float | None | tuple[float | None, int]:
    """
    Compute average PSNR over all frames.
    Expects `pred` and `target` in shape [3, T, H, W] with values in [0, 1].
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")

    if pred.ndim != 4 or pred.shape[0] != 3:
        raise ValueError(f"Expected [3, T, H, W], got {tuple(pred.shape)}")

    valid = _valid_frame_mask(
        valid_frames,
        pred.shape[1],
        exclude_first_frame=exclude_first_frame,
    )
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return (None, 0) if return_count else None

    pred = pred.float()
    target = target.float()
    mse = (pred - target).pow(2).mean(dim=(0, 2, 3))  # [T]
    psnr = 10.0 * torch.log10((data_range * data_range) / (mse + eps))
    value = float(psnr[valid.to(device=psnr.device)].mean().item())
    return (value, valid_count) if return_count else value


def video_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    *,
    valid_frames: Optional[torch.Tensor] = None,
    exclude_first_frame: bool = False,
    return_count: bool = False,
) -> float | None | tuple[float | None, int]:
    """
    Compute average SSIM over all frames.
    Expects `pred` and `target` in shape [3, T, H, W] with values in [0, 1].
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.ndim != 4 or pred.shape[0] != 3:
        raise ValueError(f"Expected [3, T, H, W], got {tuple(pred.shape)}")
    if kernel_size % 2 == 0:
        raise ValueError("`kernel_size` must be odd.")

    valid = _valid_frame_mask(
        valid_frames,
        pred.shape[1],
        exclude_first_frame=exclude_first_frame,
    )
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return (None, 0) if return_count else None

    pred = pred.float().permute(1, 0, 2, 3).contiguous()  # [T, 3, H, W]
    target = target.float().permute(1, 0, 2, 3).contiguous()  # [T, 3, H, W]
    channels = pred.shape[1]
    kernel = _gaussian_kernel_2d(
        kernel_size=kernel_size,
        sigma=sigma,
        channels=channels,
        device=pred.device,
        dtype=pred.dtype,
    )

    pad = kernel_size // 2
    mu_x = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-12)
    per_frame = ssim_map.mean(dim=(1, 2, 3))
    value = float(per_frame[valid.to(device=per_frame.device)].mean().item())
    return (value, valid_count) if return_count else value


def video_metric_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_frames: Optional[torch.Tensor] = None,
    exclude_first_frame: bool = False,
) -> tuple[float | None, float | None, int]:
    """Compute PSNR and SSIM with one shared valid-frame denominator."""
    psnr, valid_count = video_psnr(
        pred,
        target,
        valid_frames=valid_frames,
        exclude_first_frame=exclude_first_frame,
        return_count=True,
    )
    ssim = video_ssim(
        pred,
        target,
        valid_frames=valid_frames,
        exclude_first_frame=exclude_first_frame,
    )
    return psnr, ssim, valid_count


def masked_action_errors(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    action_is_pad: Optional[torch.Tensor] = None,
    action_dim_is_pad: Optional[torch.Tensor] = None,
    return_count: bool = False,
) -> tuple[float | None, float | None, int] | tuple[float | None, float | None]:
    """Reduce action errors over explicitly valid time/dimension elements.

    ``pred`` and ``target`` may be ``[T, D]`` or a single ``[1, T, D]``
    request.  Action dimension masks accept ``[D]``, ``[T, D]``, ``[1, D]``
    or ``[1, T, D]``; these forms are kept explicit so a malformed mask cannot
    silently broadcast over a different batch or horizon.
    """
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("`pred` and `target` must be torch.Tensor instances.")
    if pred.shape != target.shape:
        raise ValueError(
            f"Action shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    if pred.ndim == 3:
        if pred.shape[0] != 1:
            raise ValueError(
                "Action metrics support [T, D] or a single [1, T, D] request, "
                f"got {tuple(pred.shape)}."
            )
        pred = pred[0]
        target = target[0]
    if pred.ndim != 2:
        raise ValueError(f"Expected action shape [T, D] or [1, T, D], got {tuple(pred.shape)}")

    num_steps, num_dims = pred.shape
    valid = torch.ones((num_steps, num_dims), dtype=torch.bool, device=pred.device)

    def _temporal_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`{name}` must be a torch.Tensor or None, got {type(mask)}.")
        if mask.ndim == 2:
            if mask.shape[0] != 1:
                raise ValueError(
                    f"`{name}` must have shape [T] or [1, T], got {tuple(mask.shape)}."
                )
            mask = mask[0]
        if mask.ndim != 1 or mask.shape[0] != num_steps:
            raise ValueError(
                f"`{name}` must have shape [T] (or [1, T]), "
                f"got {tuple(mask.shape)} for T={num_steps}."
            )
        return mask.to(device=pred.device, dtype=torch.bool)

    def _dimension_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`{name}` must be a torch.Tensor or None, got {type(mask)}.")
        if mask.ndim == 1:
            if mask.shape[0] != num_dims:
                raise ValueError(
                    f"`{name}` must have last dimension D={num_dims}, got {tuple(mask.shape)}."
                )
            return mask.to(device=pred.device, dtype=torch.bool).unsqueeze(0).expand(num_steps, -1)
        if mask.ndim == 2:
            if tuple(mask.shape) == (1, num_dims):
                return mask.to(device=pred.device, dtype=torch.bool).expand(num_steps, -1)
            if tuple(mask.shape) == (num_steps, num_dims):
                return mask.to(device=pred.device, dtype=torch.bool)
            raise ValueError(
                f"`{name}` must have shape [D], [1, D], or [T, D], "
                f"got {tuple(mask.shape)} for T={num_steps}, D={num_dims}."
            )
        if mask.ndim == 3 and tuple(mask.shape) == (1, num_steps, num_dims):
            return mask[0].to(device=pred.device, dtype=torch.bool)
        raise ValueError(
            f"`{name}` must have shape [D], [1, D], [T, D], or [1, T, D], "
            f"got {tuple(mask.shape)}."
        )

    if action_is_pad is not None:
        valid &= ~_temporal_mask(action_is_pad, "action_is_pad").unsqueeze(1)
    if action_dim_is_pad is not None:
        valid &= ~_dimension_mask(action_dim_is_pad, "action_dim_is_pad")

    valid_count = int(valid.sum().item())
    if valid_count == 0:
        result = (None, None, 0)
    else:
        diff = (pred.float() - target.float()).abs()
        l1 = float(diff.masked_select(valid).mean().item())
        l2 = float(diff.square().masked_select(valid).mean().item())
        result = (l1, l2, valid_count)
    return result if return_count else result[:2]
