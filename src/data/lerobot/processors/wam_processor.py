from collections.abc import Mapping
from typing import Dict, Any, Optional, List, Tuple, Sequence

import torch
import numpy as np
from copy import deepcopy
from ..utils.normalizer import LinearNormalizer, NormMode
from utils.logging_config import get_logger
from .base_processor import BaseProcessor

logger = get_logger(__name__)

class WAMProcessor(BaseProcessor):
    def __init__(
        self,
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int, 
        action_output_dim: int,
        proprio_output_dim: int,

        action_state_transforms: Optional[List[Any]], 

        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        norm_exception_mode: Dict[str, Dict[str, NormMode]], 

        action_state_merger, 

        train_transforms: Dict[str, List[Any]] | None,
        val_transforms: Dict[str, List[Any]] | None, 

        drop_high_level_prob: float = 1.0,
        use_zh_instruction: bool = False,

        tokenizer: Optional[Any] = None,
        delta_action_dim_mask: Optional[Dict[str, List[bool]]] = None,
        action_clip_range: Optional[Tuple[float, float]] = (-5.0, 5.0),
        state_clip_range: Optional[Tuple[float, float]] = (-5.0, 5.0),
        binary_dims: Optional[Dict[str, Dict[str, Sequence[int]]]] = None,
        require_raw_state_anchor: bool = False,
    ):
        self.shape_meta = shape_meta
        self.num_obs_steps = num_obs_steps
        self.num_output_cameras = num_output_cameras
        self.action_output_dim = action_output_dim
        self.proprio_output_dim = proprio_output_dim

        self.drop_high_level_prob = drop_high_level_prob
        self.use_zh_instruction = use_zh_instruction

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        self._is_train = None

        self.action_state_transforms = action_state_transforms
        self.action_state_merger = action_state_merger
        self.action_state_merger.set_shape_meta(self.shape_meta)

        self.use_stepwise_action_norm = use_stepwise_action_norm
        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        # Opt out only for new recipes; legacy configs retain clamp5 behavior.
        self.action_clip_range = action_clip_range
        # State clipping remains legacy-compatible unless a new recipe opts out.
        self.state_clip_range = state_clip_range
        self.binary_dims = binary_dims
        if binary_dims is not None:
            # Validate the schema at config construction time as well as when
            # stats are installed, so a malformed recipe fails closed early.
            LinearNormalizer._validate_binary_dims(self.shape_meta, binary_dims)
        if not isinstance(require_raw_state_anchor, bool):
            raise TypeError("require_raw_state_anchor must be a boolean")
        self.require_raw_state_anchor = require_raw_state_anchor
        self._normalizer = None

        self.tokenizer = tokenizer
        if delta_action_dim_mask is None:
            self.delta_action_dim_mask = None
        else:
            action_meta = self.shape_meta["action"]
            expected_keys = [m["key"] for m in action_meta]
            provided_keys = list(delta_action_dim_mask.keys())
            if set(provided_keys) != set(expected_keys):
                raise ValueError(
                    f"`delta_action_dim_mask` keys mismatch. Expected {expected_keys}, got {provided_keys}."
                )

            self.delta_action_dim_mask = {}
            for meta in action_meta:
                key = meta["key"]
                expected_dim = meta["shape"]
                mask = delta_action_dim_mask[key]
                if len(mask) != expected_dim:
                    raise ValueError(
                        f"`delta_action_dim_mask[{key}]` length must be {expected_dim}, got {len(mask)}."
                    )
                self.delta_action_dim_mask[key] = torch.as_tensor(mask, dtype=torch.bool)

    @property
    def is_train(self):
        if self._is_train is None:
            raise ValueError("is_train has not been set. Please call train() and eval() first.")
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        if self._normalizer is None:
            raise ValueError("normalizer has not been set. Please call set_normalizer_from_stats() first.")
        return self._normalizer

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any] = None):
        self._normalizer = LinearNormalizer(
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            shape_meta=self.shape_meta,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
            action_clip_range=self.action_clip_range,
            state_clip_range=self.state_clip_range,
            binary_dims=self.binary_dims,
        )

    @staticmethod
    def _clone_state(
        state: Dict[str, torch.Tensor], *, add_batch_dim: bool = False, first_step_only: bool = False
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(state, Mapping):
            raise TypeError(f"state must be a mapping of field keys to tensors, got {type(state)!r}")
        cloned = {}
        for key, value in state.items():
            value = value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
            if first_step_only and isinstance(value, torch.Tensor) and value.ndim >= 2:
                value = value[..., :1, :]
            if add_batch_dim and isinstance(value, torch.Tensor) and value.ndim == 2:
                value = value.unsqueeze(0)
            cloned[key] = value
        return cloned

    def _state_dim(self) -> int:
        if len(self.shape_meta["state"]) != 1:
            raise ValueError("Raw-state tensor input requires exactly one state field")
        shape = self.shape_meta["state"][0]["shape"]
        if isinstance(shape, int):
            return shape
        if isinstance(shape, Sequence) and len(shape) == 1:
            return int(shape[0])
        raise ValueError(f"Raw-state tensor input requires a one-dimensional state field, got {shape!r}")

    def _binary_state_dims(self, key: str) -> Tuple[int, ...]:
        if not isinstance(self.binary_dims, Mapping):
            return ()
        state_dims = self.binary_dims.get("state", {})
        if not isinstance(state_dims, Mapping):
            return ()
        dims = state_dims.get(key, ())
        return tuple(int(dim) for dim in dims)

    def _coerce_raw_field(
        self,
        value,
        *,
        name: str,
        binary_key: str,
        dim: int,
        action_btd: torch.Tensor,
        action_was_unbatched: bool,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.ndim == 0:
            raise ValueError(f"{name} must be [D], [B,D], [T,D], or [B,T,D], not a scalar")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        if value.shape[-1] != dim:
            raise ValueError(f"{name} must end in state dimension {dim}, got {tuple(value.shape)}")

        batch_size, action_horizon = action_btd.shape[:2]
        if value.ndim == 1:
            if batch_size != 1:
                raise ValueError(f"{name} batch mismatch: scalar state cannot anchor batch size {batch_size}")
            value = value.reshape(1, 1, dim)
        elif value.ndim == 2:
            length = value.shape[0]
            if action_was_unbatched and length in (action_horizon, self.num_obs_steps):
                value = value.unsqueeze(0)
            elif length == batch_size:
                value = value.unsqueeze(1)
            elif batch_size == 1 and length in (action_horizon, self.num_obs_steps):
                value = value.unsqueeze(0)
            else:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} does not match action batch/horizon "
                    f"({batch_size}, {action_horizon}); provide [1,T,D] for an unbatched sequence"
                )
        elif value.ndim == 3:
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch mismatch: expected {batch_size}, got {value.shape[0]}"
                )
            if value.shape[1] not in (1, action_horizon, self.num_obs_steps):
                raise ValueError(
                    f"{name} time dimension must be 1, {action_horizon}, or {self.num_obs_steps}, "
                    f"got {value.shape[1]}"
                )
        else:
            raise ValueError(f"{name} must be [D], [B,D], [T,D], or [B,T,D], got {tuple(value.shape)}")

        binary_dims = self._binary_state_dims(binary_key)
        if binary_dims:
            binary_values = value[..., list(binary_dims)]
            if not torch.logical_or(binary_values == 0, binary_values == 1).all():
                raise ValueError(f"{name} binary dimensions must contain only exact 0/1 values")
        return value.to(device=reference.device, dtype=reference.dtype)

    def _raw_state_mapping(
        self,
        raw_state,
        *,
        reference_state: Dict[str, torch.Tensor],
        action_btd: torch.Tensor,
        action_was_unbatched: bool,
    ) -> Dict[str, torch.Tensor]:
        """Convert an explicit raw anchor to the unmerged [B,T,D] state mapping."""
        if isinstance(raw_state, Mapping):
            expected_keys = {meta["key"] for meta in self.shape_meta["state"]}
            unknown = set(raw_state) - expected_keys
            missing = expected_keys - set(raw_state)
            if unknown or missing:
                raise ValueError(
                    f"raw_state keys must match state fields; missing={sorted(missing)!r}, "
                    f"unknown={sorted(unknown)!r}"
                )
            mapping = {}
            for meta in self.shape_meta["state"]:
                key = meta["key"]
                shape = meta["shape"]
                dim = int(shape) if isinstance(shape, int) else int(shape[0])
                mapping[key] = self._coerce_raw_field(
                    raw_state[key],
                    name=f"raw_state[{key!r}]",
                    binary_key=key,
                    dim=dim,
                    action_btd=action_btd,
                    action_was_unbatched=action_was_unbatched,
                    reference=reference_state[key],
                )
            return mapping

        if len(self.shape_meta["state"]) != 1:
            raise ValueError("Raw-state tensor input requires exactly one state field; provide a mapping otherwise")
        state_dim = self._state_dim()
        key = self.shape_meta["state"][0]["key"]
        raw_btd = self._coerce_raw_field(
            raw_state,
            name="raw_state",
            binary_key=key,
            dim=state_dim,
            action_btd=action_btd,
            action_was_unbatched=action_was_unbatched,
            reference=reference_state[key],
        )
        # A single state field needs no merger round trip. This also avoids
        # requiring a dummy action tensor for mergers whose backward method
        # unconditionally decodes both modalities.
        return {key: raw_btd}

    def augment_instruction(self, data: Dict[str, str] | List[str]) -> List[str]:
        """Choose the instruction language and optional high-level task."""
        if "coarse_task" in data:
            high_level_instruction = data["coarse_task"]
        else:
            high_level_instruction = ""
        if "task" not in data:
            return f"[high] {high_level_instruction}"

        low_level_instruction = data["task"]
        # Galaxea lerobot use @ to split Chinese and English instruction
        if "@" in low_level_instruction:
            zh, eng = low_level_instruction.split("@")
            low_level_instruction = zh if self.use_zh_instruction else eng

        if np.random.rand() < self.drop_high_level_prob:
            instruction = low_level_instruction
        else: 
            instruction = f"[High]: {high_level_instruction}, [Low]: {low_level_instruction}"
        
        return instruction

    def action_state_transform(self, batch):
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["raw_shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, \
                    f"Action key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
                    
        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, \
                f"State key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
        
        if self.action_state_transforms is not None: 
            for trans in self.action_state_transforms:
                batch = trans.forward(batch)
        
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, \
                    f"Action key {k} actual transformed shape {actual_shape} mismatch with meta shape {meta_shape}."
        
        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, \
                f"State key {k} actual transformed shape {actual_shape} mismatch with meta raw shape {meta_shape}."
        
        return batch

    def preprocess(self, data: Dict[str, Any], *, return_raw_state: bool = False) -> Dict[str, Any]:
        """Transform a dataset sample into the tensors consumed by WAM."""
        sample = {}
        sample["instruction"] = self.augment_instruction(data)
        sample["image_is_pad"] = data["image_is_pad"]
        if sample["image_is_pad"].ndim != 1:
            raise ValueError(
                "`image_is_pad` must be one-dimensional, "
                f"got shape {tuple(sample['image_is_pad'].shape)}."
            )
        num_image_steps = int(sample["image_is_pad"].shape[0])

        processed_images = []
        for meta in self.shape_meta["images"]:
            key, shape = meta["key"], meta["shape"]
            image = data["images"][key]
            assert image.ndim == 4, f"Expected 4 dimensions (num_image_steps, C, H, W), got shape {image.shape}"
            
            transforms = self.train_transforms if self.is_train else self.val_transforms
            current_transforms = transforms[key] if isinstance(transforms, dict) else transforms
            for trans in current_transforms:
                image = trans(image)
            
            meta_shape = [num_image_steps] + shape
            assert tuple(image.shape) == tuple(meta_shape), \
                f"Expected shape {meta_shape}, got {image.shape} after transforms for key {key}"

            processed_images.append(image)
        pixel_values = torch.stack(processed_images, dim=0)
        
        if self.num_output_cameras > pixel_values.shape[0]:
            out = torch.zeros((self.num_output_cameras,) + pixel_values.shape[1:], device=pixel_values.device, dtype=pixel_values.dtype)
            out[0: pixel_values.shape[0]] = pixel_values
            sample["pixel_values"] = out
        elif self.num_output_cameras < pixel_values.shape[0]:
            logger.warning(f"num_output_cameras {self.num_output_cameras} is less than the number of cameras in data {pixel_values.shape[0]}, "
                           f"truncating the input to the first {self.num_output_cameras} cameras.")
            sample["pixel_values"] = pixel_values[:self.num_output_cameras]
        else:
            sample["pixel_values"] = pixel_values

        # Keep ground-truth actions for open-loop evaluation.
        if not self.is_train and "action" in data:
            sample["gt_action"] = deepcopy(data["action"])

        if "action" in data and self.delta_action_dim_mask is not None:
            action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
            for key, dim_mask in self.delta_action_dim_mask.items():
                cur_action = data["action"][key]
                cur_action_is_pad = action_is_pad.to(device=cur_action.device)
                cur_dim_mask = dim_mask.to(device=cur_action.device)
                pad_delta_mask = cur_action_is_pad.unsqueeze(1) & cur_dim_mask.unsqueeze(0)
                cur_action[pad_delta_mask] = 0.0
        data = self.action_state_transform(data)
        # The relative transform only needs q0. Keep that anchor separate and
        # explicit as [B,1,D], while the full normalized proprio remains in the
        # ordinary model input.
        raw_state = (
            self._clone_state(data["state"], add_batch_dim=True, first_step_only=True)
            if return_raw_state
            else None
        )
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        if "action" in data:
            sample["action"] = data["action"]
            sample["action_is_pad"] = data["action_is_pad"]
            sample["action_dim_is_pad"] = data["action_dim_is_pad"]
            assert sample["action"].shape[-1] == self.action_output_dim
        sample["proprio"] = data["state"]
        sample["proprio_is_pad"] = data["state_is_pad"]
        sample["proprio_dim_is_pad"] = data["state_dim_is_pad"]
        assert sample["proprio"].shape[-1] == self.proprio_output_dim

        sample["idx"] = data["idx"]

        # This is deliberately opt-in: callers must extract this side channel
        # before constructing model kwargs rather than forwarding it blindly.
        if raw_state is not None:
            sample["raw_state"] = raw_state

        return sample

    def postprocess(self, data: Dict[str, Any], raw_state=None) -> Dict[str, Any]:
        """Undo action transforms and normalization for policy output."""
        assert "action" in data, "Action is required in postprocess"
        if self.require_raw_state_anchor and raw_state is None:
            raise ValueError(
                "This processor requires an explicit raw_state anchor in postprocess(); "
                "do not use clipped proprio as the relative-action anchor."
            )
        if "proprio" not in data:
            raise ValueError("Proprio is required in postprocess")

        action = data["action"]
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action)
        action_was_unbatched = action.ndim in (1, 2)
        if action.ndim == 1:
            action = action.reshape(1, 1, -1)
        elif action.ndim == 2:
            action = action.unsqueeze(0)
        elif action.ndim != 3:
            raise ValueError(f"action must be [D], [T,D], or [B,T,D], got {tuple(action.shape)}")
        data["action"] = action

        proprio = data["proprio"]
        if not isinstance(proprio, torch.Tensor):
            proprio = torch.as_tensor(proprio)
        if proprio.ndim == 0:
            raise ValueError("proprio must have a final state dimension, not be a scalar")
        merged_dim = proprio.shape[-1]
        if proprio.ndim == 1:
            if action.shape[0] != 1:
                raise ValueError(
                    f"proprio batch mismatch: a single state cannot anchor batch size {action.shape[0]}"
                )
            proprio = proprio.reshape(1, 1, merged_dim)
        elif proprio.ndim == 2:
            length = proprio.shape[0]
            valid_sequence_lengths = (action.shape[1], self.num_obs_steps)
            if action_was_unbatched and length in valid_sequence_lengths:
                proprio = proprio.unsqueeze(0)
            elif length == action.shape[0]:
                proprio = proprio.unsqueeze(1)
            elif action.shape[0] == 1 and length in valid_sequence_lengths:
                proprio = proprio.unsqueeze(0)
            else:
                raise ValueError(
                    f"proprio shape {tuple(proprio.shape)} does not match action batch/horizon "
                    f"({action.shape[0]}, {action.shape[1]}) or num_obs_steps={self.num_obs_steps}; "
                    "provide [B,1,D] or [B,T,D] explicitly"
                )
        elif proprio.ndim != 3:
            raise ValueError(f"proprio must be [D], [B,D], or [B,T,D], got {tuple(proprio.shape)}")
        else:
            if proprio.shape[0] != action.shape[0]:
                raise ValueError(
                    f"proprio batch mismatch: expected {action.shape[0]}, got {proprio.shape[0]}"
                )
            if proprio.shape[1] not in (1, action.shape[1], self.num_obs_steps):
                raise ValueError(
                    f"proprio time dimension must be 1, {action.shape[1]}, or {self.num_obs_steps}, "
                    f"got {proprio.shape[1]}"
                )
        data["proprio"] = proprio

        data["state"] = data.pop("proprio")
        data = self.action_state_merger.backward(data)
        data = self.normalizer.backward(data)
        decoded_state = data["state"]
        if raw_state is not None:
            data["state"] = self._raw_state_mapping(
                raw_state,
                reference_state=data["state"],
                action_btd=action,
                action_was_unbatched=action_was_unbatched,
            )
        if self.action_state_transforms is not None:
            for trans in reversed(self.action_state_transforms):
                data = trans.backward(data)
        if raw_state is not None:
            # raw_state is an explicit relative-action anchor only. Preserve
            # the normal postprocess state payload and its original horizon.
            data["state"] = decoded_state

        # WAM predicts from the current frame, not after a past-observation prefix.
        return data
