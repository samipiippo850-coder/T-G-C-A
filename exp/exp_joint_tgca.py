from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from exp.exp_long_term_forecasting import (
    Exp_Long_Term_Forecast,
)


MODULE_TOKENS = (
    "tgca",
    "adapter",
    "gate",
    "direction",
    "residual",
    "spectral",
    "cov",
    "relax",
    "curv",
)

HEAD_TOKENS = (
    "projection",
    "projector",
    "predict_linear",
    "output_layer",
    "out_proj",
    "forecast_head",
    ".head",
)


class JointMSEMAELoss(nn.Module):
    def __init__(
        self,
        experiment: "Exp_Joint_TGCA",
    ) -> None:
        super().__init__()

        self.experiment = experiment

        self.mse_weight = float(
            os.environ.get(
                "JOINT_MSE_WEIGHT",
                "0.60",
            )
        )

        self.mae_weight = float(
            os.environ.get(
                "JOINT_MAE_WEIGHT",
                "0.40",
            )
        )

        self.module_anchor_weight = float(
            os.environ.get(
                "JOINT_MODULE_ANCHOR",
                "0.0001",
            )
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        mse = torch.mean(
            (prediction - target) ** 2
        )

        mae = torch.mean(
            torch.abs(prediction - target)
        )

        loss = (
            self.mse_weight * mse
            + self.mae_weight * mae
        )

        if (
            self.module_anchor_weight > 0.0
            and self.experiment._joint_anchors
        ):
            anchor_loss = prediction.new_zeros(())

            core = self.experiment._joint_core_model()

            for name, parameter in (
                core.named_parameters()
            ):
                reference = (
                    self.experiment
                    ._joint_anchors
                    .get(name)
                )

                if (
                    reference is None
                    or not parameter.requires_grad
                ):
                    continue

                anchor_loss = (
                    anchor_loss
                    + torch.mean(
                        (
                            parameter
                            - reference
                        ) ** 2
                    )
                )

            loss = (
                loss
                + self.module_anchor_weight
                * anchor_loss
            )

        return loss


class Exp_Joint_TGCA(
    Exp_Long_Term_Forecast
):
    """
    Joint TGCA/backbone fine-tuning experiment.

    Stage A:
        TGCA/module parameters only.

    Stage B:
        module + output head + last 15% of
        remaining backbone parameter tensors.

    Stage C:
        full joint fine-tuning with discriminative
        learning rates.
    """

    def __init__(self, args) -> None:
        self.joint_stage = (
            os.environ.get(
                "JOINT_STAGE",
                "A",
            )
            .strip()
            .upper()
        )

        if self.joint_stage not in {
            "A",
            "B",
            "C",
        }:
            raise ValueError(
                "JOINT_STAGE must be A, B or C"
            )

        self._joint_checkpoint_loaded = False
        self._joint_anchors: Dict[
            str,
            torch.Tensor,
        ] = {}

        self._joint_tail_names: set[str] = set()

        super().__init__(args)

        self._joint_configure_trainable()

    def _joint_core_model(self) -> nn.Module:
        if isinstance(
            self.model,
            nn.DataParallel,
        ):
            return self.model.module

        return self.model

    @staticmethod
    def _joint_parameter_kind(
        name: str,
    ) -> str:
        lower = name.lower()

        if any(
            token in lower
            for token in MODULE_TOKENS
        ):
            return "module"

        if any(
            token in lower
            for token in HEAD_TOKENS
        ):
            return "head"

        return "backbone"

    def _joint_configure_trainable(
        self,
    ) -> None:
        core = self._joint_core_model()

        named = list(
            core.named_parameters()
        )

        backbone_names = [
            name
            for name, _ in named
            if self._joint_parameter_kind(
                name
            ) == "backbone"
        ]

        tail_fraction = float(
            os.environ.get(
                "JOINT_TAIL_FRACTION",
                "0.15",
            )
        )

        tail_count = max(
            1,
            int(
                math.ceil(
                    len(backbone_names)
                    * tail_fraction
                )
            ),
        )

        self._joint_tail_names = set(
            backbone_names[-tail_count:]
        )

        counts = {
            "module": 0,
            "head": 0,
            "tail": 0,
            "backbone": 0,
        }

        trainable_names: List[str] = []

        for name, parameter in named:
            kind = self._joint_parameter_kind(
                name
            )

            if self.joint_stage == "A":
                trainable = kind == "module"

            elif self.joint_stage == "B":
                trainable = (
                    kind in {"module", "head"}
                    or name
                    in self._joint_tail_names
                )

            else:
                trainable = True

            parameter.requires_grad = trainable

            if trainable:
                trainable_names.append(name)

                if kind == "module":
                    counts["module"] += (
                        parameter.numel()
                    )
                elif kind == "head":
                    counts["head"] += (
                        parameter.numel()
                    )
                elif (
                    name
                    in self._joint_tail_names
                ):
                    counts["tail"] += (
                        parameter.numel()
                    )
                else:
                    counts["backbone"] += (
                        parameter.numel()
                    )

        if not trainable_names:
            all_names = [
                name for name, _ in named
            ]

            raise RuntimeError(
                "No trainable parameters found. "
                "Parameter names:\n"
                + "\n".join(all_names)
            )

        total_trainable = sum(
            parameter.numel()
            for _, parameter in named
            if parameter.requires_grad
        )

        total_parameters = sum(
            parameter.numel()
            for _, parameter in named
        )

        print(
            "[JOINT_TRAINABLE]",
            {
                "stage": self.joint_stage,
                "trainable": total_trainable,
                "total": total_parameters,
                "ratio": (
                    total_trainable
                    / max(total_parameters, 1)
                ),
                "groups": counts,
            },
            flush=True,
        )

        print(
            "[JOINT_TRAINABLE_NAMES]",
            trainable_names[:100],
            flush=True,
        )

    def _joint_load_initial_checkpoint(
        self,
    ) -> None:
        if self._joint_checkpoint_loaded:
            return

        checkpoint_text = os.environ.get(
            "JOINT_INIT_CKPT",
            "",
        ).strip()

        if not checkpoint_text:
            raise RuntimeError(
                "JOINT_INIT_CKPT is required "
                "for joint fine-tuning"
            )

        checkpoint_path = Path(
            checkpoint_text
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                checkpoint_path
            )

        try:
            state = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            state = torch.load(
                checkpoint_path,
                map_location=self.device,
            )

        if isinstance(state, dict):
            for key in (
                "model_state_dict",
                "state_dict",
                "model",
            ):
                candidate = state.get(key)

                if isinstance(
                    candidate,
                    dict,
                ):
                    state = candidate
                    break

        if not isinstance(state, dict):
            raise TypeError(
                "Checkpoint does not contain "
                "a state_dict"
            )

        if state and all(
            str(key).startswith("module.")
            for key in state
        ):
            state = {
                str(key)[len("module."):]:
                    value
                for key, value in state.items()
            }

        core = self._joint_core_model()

        missing, unexpected = (
            core.load_state_dict(
                state,
                strict=False,
            )
        )

        print(
            "[JOINT_INIT_CKPT]",
            {
                "path": str(checkpoint_path),
                "missing": list(missing),
                "unexpected":
                    list(unexpected),
            },
            flush=True,
        )

        if len(missing) > 0:
            raise RuntimeError(
                "Initial checkpoint has missing "
                f"parameters: {missing}"
            )

        if len(unexpected) > 0:
            raise RuntimeError(
                "Initial checkpoint has unexpected "
                f"parameters: {unexpected}"
            )

        self._joint_checkpoint_loaded = True

    def _joint_snapshot_module_anchors(
        self,
    ) -> None:
        core = self._joint_core_model()

        self._joint_anchors = {
            name: parameter.detach().clone()
            for name, parameter
            in core.named_parameters()
            if (
                parameter.requires_grad
                and self._joint_parameter_kind(
                    name
                ) == "module"
            )
        }

        print(
            "[JOINT_ANCHOR_SNAPSHOT]",
            {
                "parameter_tensors":
                    len(self._joint_anchors),
                "weight": float(
                    os.environ.get(
                        "JOINT_MODULE_ANCHOR",
                        "0.0001",
                    )
                ),
            },
            flush=True,
        )

    def _select_criterion(self):
        return JointMSEMAELoss(self)

    def _select_optimizer(self):
        core = self._joint_core_model()

        module_lr = float(
            os.environ.get(
                "JOINT_MODULE_LR",
                "0.0001",
            )
        )

        head_lr = float(
            os.environ.get(
                "JOINT_HEAD_LR",
                "0.00002",
            )
        )

        tail_lr = float(
            os.environ.get(
                "JOINT_TAIL_LR",
                str(head_lr),
            )
        )

        backbone_lr = float(
            os.environ.get(
                "JOINT_BACKBONE_LR",
                "0.000005",
            )
        )

        weight_decay = float(
            os.environ.get(
                "JOINT_WEIGHT_DECAY",
                "0.0001",
            )
        )

        groups: Dict[
            str,
            List[torch.nn.Parameter],
        ] = {
            "module": [],
            "head": [],
            "tail": [],
            "backbone": [],
        }

        for name, parameter in (
            core.named_parameters()
        ):
            if not parameter.requires_grad:
                continue

            kind = self._joint_parameter_kind(
                name
            )

            if kind == "module":
                groups["module"].append(
                    parameter
                )

            elif kind == "head":
                groups["head"].append(
                    parameter
                )

            elif name in self._joint_tail_names:
                groups["tail"].append(
                    parameter
                )

            else:
                groups["backbone"].append(
                    parameter
                )

        lr_map = {
            "module": module_lr,
            "head": head_lr,
            "tail": tail_lr,
            "backbone": backbone_lr,
        }

        optimizer_groups = []

        report = {}

        for name in (
            "module",
            "head",
            "tail",
            "backbone",
        ):
            parameters = groups[name]

            if not parameters:
                continue

            optimizer_groups.append(
                {
                    "params": parameters,
                    "lr": lr_map[name],
                    "weight_decay":
                        weight_decay,
                    "joint_group": name,
                }
            )

            report[name] = {
                "lr": lr_map[name],
                "parameter_tensors":
                    len(parameters),
                "parameters": sum(
                    parameter.numel()
                    for parameter
                    in parameters
                ),
            }

        if not optimizer_groups:
            raise RuntimeError(
                "No optimizer parameter groups"
            )

        print(
            "[JOINT_OPTIMIZER_GROUPS]",
            report,
            flush=True,
        )

        return torch.optim.AdamW(
            optimizer_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def train(self, setting):
        self._joint_load_initial_checkpoint()
        self._joint_snapshot_module_anchors()

        return super().train(setting)
