from __future__ import annotations

import atexit
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def _first_tensor(obj: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, (list, tuple)):
        for item in obj:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor

    if isinstance(obj, dict):
        for item in obj.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor

    return None


class LatentAuditHook:
    """
    非侵入式导出某个中间模块的输出。

    最终统一保存为：
        [N, T, D]

    如果原始输出是：
        [B, D]       -> [B, 1, D]
        [B, T, D]    -> 保持不变
        更高维       -> [B, -1, last_dim]
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.pattern = os.getenv(
            "TGCA_LATENT_HOOK_PATTERN",
            r"core\.state\.state_mlp",
        )

        matches = [
            (name, module)
            for name, module in model.named_modules()
            if re.search(self.pattern, name)
        ]

        if not matches:
            candidates = [
                name
                for name, _ in model.named_modules()
                if any(
                    token in name.lower()
                    for token in (
                        "core",
                        "state",
                        "latent",
                        "encoder",
                        "bank",
                        "hidden",
                    )
                )
            ]

            raise RuntimeError(
                f"[TGCA_AUDIT] no module matched pattern={self.pattern!r}\n"
                "Candidate module names:\n"
                + "\n".join(candidates[:150])
            )

        # 优先完全匹配，然后选择路径最短的模块。
        matches.sort(
            key=lambda item: (
                item[0] != self.pattern,
                len(item[0]),
                item[0],
            )
        )

        self.module_name, module = matches[0]

        self.out_dir = Path(
            os.getenv(
                "TGCA_LATENT_AUDIT_OUT",
                "experiments/"
                "latent_spectral_dissipation_audit_v1/raw/default",
            )
        )

        self.max_samples = int(
            os.getenv("TGCA_LATENT_MAX_SAMPLES", "256")
        )
        self.max_time = int(
            os.getenv("TGCA_LATENT_MAX_TIME", "96")
        )
        self.max_dim = int(
            os.getenv("TGCA_LATENT_MAX_DIM", "128")
        )

        # 某些模块可能输出 [B,D,T]，此时设为 1。
        self.transpose_time_feature = (
            os.getenv("TGCA_LATENT_TRANSPOSE", "0") == "1"
        )

        self.chunks: list[np.ndarray] = []
        self.n_samples = 0
        self.closed = False

        self.handle = module.register_forward_hook(self._hook)
        atexit.register(self.close)

        print(
            f"[TGCA_AUDIT] attached module={self.module_name} "
            f"pattern={self.pattern!r} "
            f"out={self.out_dir}"
        )

    def _normalize_shape(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        x = tensor.detach().float().cpu()

        if x.ndim == 0:
            x = x.reshape(1, 1, 1)

        elif x.ndim == 1:
            x = x[None, None, :]

        elif x.ndim == 2:
            x = x[:, None, :]

        else:
            if self.transpose_time_feature and x.ndim == 3:
                x = x.transpose(1, 2)

            x = x.reshape(
                x.shape[0],
                -1,
                x.shape[-1],
            )

        if x.shape[1] > self.max_time:
            time_index = torch.linspace(
                0,
                x.shape[1] - 1,
                self.max_time,
            ).round().long()

            x = x.index_select(1, time_index)

        if x.shape[2] > self.max_dim:
            x = x[:, :, : self.max_dim]

        return x.contiguous()

    def _hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, inputs

        if self.closed:
            return

        if self.n_samples >= self.max_samples:
            return

        tensor = _first_tensor(output)

        if tensor is None:
            return

        x = self._normalize_shape(tensor)

        remaining = self.max_samples - self.n_samples
        x = x[:remaining]

        if x.numel() == 0:
            return

        if not self.chunks:
            print(
                "[TGCA_AUDIT] first latent batch "
                f"shape={tuple(x.shape)}"
            )

        self.chunks.append(x.numpy())
        self.n_samples += int(x.shape[0])

    def close(self) -> None:
        if self.closed:
            return

        self.closed = True

        try:
            self.handle.remove()
        except Exception:
            pass

        if not self.chunks:
            print("[TGCA_AUDIT] no latent activation captured")
            return

        self.out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        latent = np.concatenate(
            self.chunks,
            axis=0,
        )

        latent_path = self.out_dir / "latent.npy"
        np.save(latent_path, latent)

        metadata = {
            "module_name": self.module_name,
            "pattern": self.pattern,
            "latent_shape": list(latent.shape),
            "max_samples": self.max_samples,
            "max_time": self.max_time,
            "max_dim": self.max_dim,
            "transpose_time_feature": (
                self.transpose_time_feature
            ),
            "note": (
                "Latent is normalized as [N,T,D]. "
                "If T=1, representation spectra can be audited, "
                "but temporal generator/stable/drift tensors "
                "cannot yet be estimated."
            ),
        }

        (
            self.out_dir / "latent_meta.json"
        ).write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[TGCA_AUDIT] saved latent={latent.shape} "
            f"to {latent_path}"
        )


def maybe_attach_latent_audit(
    model: torch.nn.Module,
) -> Optional[LatentAuditHook]:
    if os.getenv("TGCA_LATENT_AUDIT", "0") != "1":
        return None

    existing = getattr(
        model,
        "_tgca_latent_audit_hook",
        None,
    )

    if existing is not None:
        return existing

    hook = LatentAuditHook(model)

    setattr(
        model,
        "_tgca_latent_audit_hook",
        hook,
    )

    return hook
