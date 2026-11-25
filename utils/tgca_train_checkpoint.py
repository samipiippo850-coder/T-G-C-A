from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


def _unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    """
    Support:
      - raw state_dict
      - {"state_dict": ...}
      - {"model_state_dict": ...}
      - {"model": ...}
    """
    if not isinstance(payload, dict):
        raise TypeError(
            "Checkpoint must be a dict-like object, "
            f"got {type(payload)}"
        )

    for key in [
        "state_dict",
        "model_state_dict",
        "model",
    ]:
        value = payload.get(key)

        if isinstance(value, dict):
            return value

    # Raw state_dict: all or most values are tensors.
    tensor_values = [
        value
        for value in payload.values()
        if torch.is_tensor(value)
    ]

    if tensor_values:
        return payload

    raise RuntimeError(
        "Could not find a model state_dict in checkpoint. "
        f"Top-level keys: {list(payload.keys())[:30]}"
    )


def _clean_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    cleaned = {}

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        new_key = str(key)

        # Common DataParallel prefix.
        while new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned[new_key] = value

    return cleaned


def _parameter_signature(
    model: nn.Module,
) -> Dict[str, float]:
    total_sum = 0.0
    total_sq = 0.0
    total_count = 0

    with torch.no_grad():
        for parameter in model.parameters():
            value = parameter.detach().float()

            total_sum += float(value.sum().cpu())
            total_sq += float(
                (value * value).sum().cpu()
            )
            total_count += int(value.numel())

    return {
        "sum": total_sum,
        "sq_sum": total_sq,
        "count": total_count,
    }


def maybe_load_train_checkpoint(
    model: nn.Module,
    device=None,
) -> nn.Module:
    """
    Load a checkpoint before training when
    TGCA_TRAIN_LOAD_CKPT is provided.

    Target policy:
      auto:
        TGCA_CPTRD -> model.base
        otherwise  -> whole model

      base:
        explicitly load model.base

      root:
        explicitly load the whole model
    """
    checkpoint_value = os.environ.get(
        "TGCA_TRAIN_LOAD_CKPT",
        "",
    ).strip()

    if not checkpoint_value:
        return model

    checkpoint_path = Path(
        checkpoint_value
    ).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "TGCA training checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    root = _unwrap_model(model)

    target_mode = os.environ.get(
        "TGCA_TRAIN_LOAD_TARGET",
        "auto",
    ).strip().lower()

    if target_mode not in {
        "auto",
        "base",
        "root",
    }:
        raise ValueError(
            "TGCA_TRAIN_LOAD_TARGET must be one of "
            "{auto, base, root}, "
            f"got {target_mode}"
        )

    if target_mode == "base":
        if not hasattr(root, "base"):
            raise AttributeError(
                "Requested target=base but model has no .base"
            )
        target = root.base
        target_name = "base"

    elif target_mode == "root":
        target = root
        target_name = "root"

    else:
        if hasattr(root, "base"):
            target = root.base
            target_name = "base"
        else:
            target = root
            target_name = "root"

    before = _parameter_signature(target)

    try:
        payload = torch.load(
            checkpoint_path,
            map_location=(
                device
                if device is not None
                else "cpu"
            ),
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(
            checkpoint_path,
            map_location=(
                device
                if device is not None
                else "cpu"
            ),
        )

    source_state = _clean_state_dict(
        _extract_state_dict(payload)
    )

    target_state = target.state_dict()

    matched = {}
    shape_mismatch = []
    unexpected = []

    for key, value in source_state.items():
        if key not in target_state:
            unexpected.append(key)
            continue

        if tuple(value.shape) != tuple(
            target_state[key].shape
        ):
            shape_mismatch.append({
                "key": key,
                "checkpoint_shape":
                    tuple(value.shape),
                "target_shape":
                    tuple(
                        target_state[key].shape
                    ),
            })
            continue

        matched[key] = value

    if not matched:
        raise RuntimeError(
            "Checkpoint loading matched zero tensors. "
            f"target={target_name}, "
            f"checkpoint={checkpoint_path}, "
            f"source_keys={list(source_state)[:20]}, "
            f"target_keys={list(target_state)[:20]}"
        )

    matched_parameter_count = sum(
        int(value.numel())
        for value in matched.values()
    )

    target_parameter_count = sum(
        int(value.numel())
        for value in target_state.values()
        if torch.is_tensor(value)
    )

    matched_ratio = (
        matched_parameter_count
        / max(target_parameter_count, 1)
    )

    minimum_ratio = float(
        os.environ.get(
            "TGCA_TRAIN_MIN_MATCH_RATIO",
            "0.90",
        )
    )

    if matched_ratio < minimum_ratio:
        raise RuntimeError(
            "Checkpoint match ratio is too low: "
            f"ratio={matched_ratio:.6f}, "
            f"minimum={minimum_ratio:.6f}, "
            f"target={target_name}, "
            f"checkpoint={checkpoint_path}"
        )

    result = target.load_state_dict(
        matched,
        strict=False,
    )

    after = _parameter_signature(target)

    root._tgca_train_checkpoint_loaded = True
    root._tgca_train_checkpoint_path = str(
        checkpoint_path
    )
    root._tgca_train_checkpoint_target = (
        target_name
    )

    print(
        "[TGCA_TRAIN_CKPT_LOADED]",
        {
            "checkpoint":
                str(checkpoint_path),
            "target":
                target_name,
            "matched_tensors":
                len(matched),
            "target_tensors":
                len(target_state),
            "matched_parameter_count":
                matched_parameter_count,
            "target_parameter_count":
                target_parameter_count,
            "matched_ratio":
                matched_ratio,
            "missing_count":
                len(result.missing_keys),
            "unexpected_count":
                len(result.unexpected_keys),
            "source_unexpected_count":
                len(unexpected),
            "shape_mismatch_count":
                len(shape_mismatch),
            "signature_before":
                before,
            "signature_after":
                after,
            "signature_changed":
                (
                    abs(
                        after["sq_sum"]
                        - before["sq_sum"]
                    )
                    > 1e-12
                ),
        },
        flush=True,
    )

    if result.missing_keys:
        print(
            "[TGCA_TRAIN_CKPT_MISSING]",
            result.missing_keys[:30],
            flush=True,
        )

    if unexpected:
        print(
            "[TGCA_TRAIN_CKPT_SOURCE_UNEXPECTED]",
            unexpected[:30],
            flush=True,
        )

    if shape_mismatch:
        print(
            "[TGCA_TRAIN_CKPT_SHAPE_MISMATCH]",
            shape_mismatch[:10],
            flush=True,
        )

    return model
