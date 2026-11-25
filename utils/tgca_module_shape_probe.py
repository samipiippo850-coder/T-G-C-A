from __future__ import annotations

import os
import re
from typing import Any

import torch


def describe_output(
    output: Any,
) -> str:
    if torch.is_tensor(output):
        return (
            f"Tensor"
            f"{tuple(output.shape)} "
            f"dtype={output.dtype}"
        )

    if isinstance(output, tuple):
        return (
            "tuple["
            + ", ".join(
                describe_output(item)
                for item in output
            )
            + "]"
        )

    if isinstance(output, list):
        return (
            "list["
            + ", ".join(
                describe_output(item)
                for item in output
            )
            + "]"
        )

    if isinstance(output, dict):
        return (
            "dict{"
            + ", ".join(
                f"{key}:"
                f"{describe_output(value)}"
                for key, value in output.items()
            )
            + "}"
        )

    return type(output).__name__


class ModuleShapeProbe:
    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:
        self.pattern = os.getenv(
            "TGCA_SHAPE_PROBE_PATTERN",
            r"^core",
        )

        self.handles = []
        self.seen = set()

        for name, module in model.named_modules():
            if not name:
                continue

            if not re.search(
                self.pattern,
                name,
            ):
                continue

            handle = module.register_forward_hook(
                self._make_hook(name)
            )

            self.handles.append(handle)

        print(
            f"[TGCA_SHAPE_PROBE] "
            f"attached={len(self.handles)} "
            f"pattern={self.pattern!r}"
        )

    def _make_hook(
        self,
        name: str,
    ):
        def hook(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            del module, inputs

            if name in self.seen:
                return

            self.seen.add(name)

            print(
                f"[TGCA_SHAPE] "
                f"{name}: "
                f"{describe_output(output)}"
            )

        return hook

    def close(self) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass

        self.handles.clear()


def maybe_attach_shape_probe(
    model: torch.nn.Module,
):
    if (
        os.getenv(
            "TGCA_SHAPE_PROBE",
            "0",
        )
        != "1"
    ):
        return None

    existing = getattr(
        model,
        "_tgca_shape_probe",
        None,
    )

    if existing is not None:
        return existing

    probe = ModuleShapeProbe(model)

    setattr(
        model,
        "_tgca_shape_probe",
        probe,
    )

    return probe
