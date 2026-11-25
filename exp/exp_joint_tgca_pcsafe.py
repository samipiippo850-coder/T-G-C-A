from __future__ import annotations

import copy
import os
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.exp_joint_tgca import Exp_Joint_TGCA


SAFETY_HEAD_NAMES = (
    "adapter.scale_logit",
)

SAFETY_HEAD_PREFIXES = (
    "adapter.gate.",
    "adapter.direction_head.",
)


def _first_tensor(value):
    if torch.is_tensor(value):
        return value

    if isinstance(value, (tuple, list)):
        for item in value:
            result = _first_tensor(item)
            if result is not None:
                return result

    if isinstance(value, dict):
        preferred = (
            "output",
            "outputs",
            "pred",
            "prediction",
            "forecast",
        )

        for key in preferred:
            if key in value:
                result = _first_tensor(value[key])
                if result is not None:
                    return result

        for item in value.values():
            result = _first_tensor(item)
            if result is not None:
                return result

    return None


class TemporalPCSafeLoss(nn.Module):
    """
    Deterministic parameter-space PCGrad over temporal
    group MSE/MAE tasks.

    Forward value:
        ordinary task loss for logging and validation.

    Backward gradient:
        projected PCGrad merge plus anchor gradient.
    """

    def __init__(
        self,
        experiment: "Exp_Joint_TGCA_PCSafe",
    ) -> None:
        super().__init__()

        self.experiment = experiment

        self.group_count = int(
            os.environ.get(
                "JOINT_PC_GROUPS",
                "2",
            )
        )

        self.gain_weight = float(
            os.environ.get(
                "JOINT_GAIN_WEIGHT",
                "2.0",
            )
        )

        self.direction_weight = float(
            os.environ.get(
                "JOINT_DIRECTION_WEIGHT",
                "1.0",
            )
        )

        self.anchor_weight = float(
            os.environ.get(
                "JOINT_MODULE_ANCHOR",
                "0.0001",
            )
        )

        self.mse_direction_margin = float(
            os.environ.get(
                "JOINT_MSE_DIRECTION_MARGIN",
                "0.0001",
            )
        )

        self.mae_direction_margin = float(
            os.environ.get(
                "JOINT_MAE_DIRECTION_MARGIN",
                "0.00001",
            )
        )

        self.gain_margin = float(
            os.environ.get(
                "JOINT_GAIN_MARGIN",
                "0.0",
            )
        )

        self.pc_eps = float(
            os.environ.get(
                "JOINT_PC_EPS",
                "1e-12",
            )
        )

        self.log_every = int(
            os.environ.get(
                "JOINT_PC_LOG_EVERY",
                "200",
            )
        )

        self.step_count = 0

    def _teacher_prediction(
        self,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        teacher = (
            self.experiment
            ._pcsafe_teacher_output
        )

        teacher = _first_tensor(teacher)

        if teacher is None:
            raise RuntimeError(
                "Teacher output was not captured "
                "during the student forward pass"
            )

        teacher = teacher.detach()

        # Mirror the usual forecast slicing.
        if teacher.ndim != prediction.ndim:
            raise RuntimeError(
                "Teacher/student ndim mismatch: "
                f"{teacher.shape} vs {prediction.shape}"
            )

        if teacher.shape[0] != prediction.shape[0]:
            raise RuntimeError(
                "Teacher/student batch mismatch: "
                f"{teacher.shape} vs {prediction.shape}"
            )

        if teacher.shape[1] != prediction.shape[1]:
            teacher = teacher[
                :,
                -prediction.shape[1]:,
                ...
            ]

        if teacher.shape[-1] != prediction.shape[-1]:
            teacher = teacher[
                ...,
                -prediction.shape[-1]:
            ]

        if teacher.shape != prediction.shape:
            raise RuntimeError(
                "Unable to align teacher output: "
                f"{teacher.shape} vs {prediction.shape}"
            )

        return teacher.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )

    def _trainable_parameters(
        self,
    ) -> List[torch.nn.Parameter]:
        core = (
            self.experiment
            ._joint_core_model()
        )

        parameters = [
            parameter
            for parameter in core.parameters()
            if parameter.requires_grad
        ]

        if not parameters:
            raise RuntimeError(
                "No trainable parameters for PCGrad"
            )

        return parameters

    def _anchor_loss(
        self,
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.anchor_weight <= 0.0
            or not self.experiment._joint_anchors
        ):
            return reference_tensor.new_zeros(())

        core = (
            self.experiment
            ._joint_core_model()
        )

        loss = reference_tensor.new_zeros(())

        for name, parameter in (
            core.named_parameters()
        ):
            if not parameter.requires_grad:
                continue

            reference = (
                self.experiment
                ._joint_anchors
                .get(name)
            )

            if reference is None:
                continue

            loss = (
                loss
                + torch.mean(
                    (parameter - reference) ** 2
                )
            )

        return loss

    @staticmethod
    def _dot(
        first: Sequence[torch.Tensor],
        second: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        result = first[0].new_zeros(())

        for left, right in zip(
            first,
            second,
        ):
            result = result + torch.sum(
                left * right
            )

        return result

    @staticmethod
    def _clone_gradient(
        gradient: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        return [
            item.clone()
            for item in gradient
        ]

    def _task_gradients(
        self,
        tasks: Sequence[torch.Tensor],
        parameters: Sequence[
            torch.nn.Parameter
        ],
    ) -> List[List[torch.Tensor]]:
        gradients = []

        for task_index, task in enumerate(tasks):
            retain_graph = (
                task_index
                < len(tasks) - 1
            )

            raw = torch.autograd.grad(
                task,
                parameters,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=True,
            )

            materialized = []

            for parameter, gradient in zip(
                parameters,
                raw,
            ):
                if gradient is None:
                    gradient = torch.zeros_like(
                        parameter
                    )

                gradient = torch.nan_to_num(
                    gradient,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                materialized.append(
                    gradient.detach()
                )

            gradients.append(materialized)

        return gradients

    def _pcgrad_merge(
        self,
        task_gradients: Sequence[
            Sequence[torch.Tensor]
        ],
    ) -> Tuple[
        List[torch.Tensor],
        int,
        int,
    ]:
        projected = []

        conflict_count = 0
        pair_count = 0

        for task_index, gradient in enumerate(
            task_gradients
        ):
            current = self._clone_gradient(
                gradient
            )

            # Fixed order for deterministic replication.
            for other_index, other in enumerate(
                task_gradients
            ):
                if other_index == task_index:
                    continue

                dot = self._dot(
                    current,
                    other,
                )

                norm = self._dot(
                    other,
                    other,
                )

                pair_count += 1

                if float(dot.detach()) < 0.0:
                    conflict_count += 1

                    coefficient = (
                        dot
                        / (
                            norm
                            + self.pc_eps
                        )
                    )

                    current = [
                        left
                        - coefficient * right
                        for left, right in zip(
                            current,
                            other,
                        )
                    ]

            projected.append(current)

        merged = []

        for parameter_index in range(
            len(projected[0])
        ):
            stacked = torch.stack(
                [
                    task[
                        parameter_index
                    ]
                    for task in projected
                ],
                dim=0,
            )

            merged.append(
                torch.mean(
                    stacked,
                    dim=0,
                )
            )

        return (
            merged,
            conflict_count,
            pair_count,
        )

    def _ordinary_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        mse = torch.mean(
            (prediction - target) ** 2
        )

        mae = torch.mean(
            torch.abs(
                prediction - target
            )
        )

        return 0.60 * mse + 0.40 * mae

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Validation and testing use the ordinary,
        # interpretable scalar objective.
        if (
            not self.training
            or not torch.is_grad_enabled()
            or not self.experiment.model.training
        ):
            return self._ordinary_loss(
                prediction,
                target,
            )

        teacher = self._teacher_prediction(
            prediction
        )

        batch_size = prediction.shape[0]

        group_count = max(
            1,
            min(
                self.group_count,
                batch_size,
            ),
        )

        indices = torch.arange(
            batch_size,
            device=prediction.device,
        )

        groups = torch.tensor_split(
            indices,
            group_count,
        )

        tasks: List[torch.Tensor] = []
        diagnostic_rows = []

        for group_index, group in enumerate(
            groups
        ):
            if group.numel() == 0:
                continue

            student_group = prediction.index_select(
                0,
                group,
            )

            teacher_group = teacher.index_select(
                0,
                group,
            )

            target_group = target.index_select(
                0,
                group,
            )

            student_error = (
                student_group - target_group
            )

            teacher_error = (
                teacher_group - target_group
            )

            direction = (
                student_group - teacher_group
            )

            student_mse = torch.mean(
                student_error ** 2
            )

            student_mae = torch.mean(
                torch.abs(student_error)
            )

            teacher_mse = torch.mean(
                teacher_error ** 2
            )

            teacher_mae = torch.mean(
                torch.abs(teacher_error)
            )

            delta_mse = (
                student_mse - teacher_mse
            )

            delta_mae = (
                student_mae - teacher_mae
            )

            mse_direction = 2.0 * torch.mean(
                teacher_error * direction
            )

            mae_direction = torch.mean(
                torch.sign(
                    teacher_error
                )
                * direction
            )

            mse_gain_penalty = F.relu(
                delta_mse
                + self.gain_margin
            )

            mae_gain_penalty = F.relu(
                delta_mae
                + self.gain_margin
            )

            mse_direction_penalty = F.relu(
                mse_direction
                + self.mse_direction_margin
            )

            mae_direction_penalty = F.relu(
                mae_direction
                + self.mae_direction_margin
            )

            mse_task = (
                student_mse
                + self.gain_weight
                * mse_gain_penalty
                + self.direction_weight
                * mse_direction_penalty
            )

            mae_task = (
                student_mae
                + self.gain_weight
                * mae_gain_penalty
                + self.direction_weight
                * mae_direction_penalty
            )

            tasks.extend(
                [mse_task, mae_task]
            )

            diagnostic_rows.append(
                {
                    "group":
                        group_index,
                    "delta_mse":
                        float(
                            delta_mse.detach()
                        ),
                    "delta_mae":
                        float(
                            delta_mae.detach()
                        ),
                    "dir_mse":
                        float(
                            mse_direction.detach()
                        ),
                    "dir_mae":
                        float(
                            mae_direction.detach()
                        ),
                }
            )

        if not tasks:
            raise RuntimeError(
                "No temporal PCGrad tasks created"
            )

        parameters = (
            self._trainable_parameters()
        )

        task_gradients = (
            self._task_gradients(
                tasks,
                parameters,
            )
        )

        (
            merged_gradient,
            conflict_count,
            pair_count,
        ) = self._pcgrad_merge(
            task_gradients
        )

        gradient_surrogate = (
            prediction.new_zeros(())
        )

        for parameter, gradient in zip(
            parameters,
            merged_gradient,
        ):
            gradient_surrogate = (
                gradient_surrogate
                + torch.sum(
                    parameter
                    * gradient.detach()
                )
            )

        anchor_loss = self._anchor_loss(
            prediction
        )

        gradient_objective = (
            gradient_surrogate
            + self.anchor_weight
            * anchor_loss
        )

        report_objective = (
            torch.mean(
                torch.stack(tasks)
            )
            + self.anchor_weight
            * anchor_loss
        )

        # Forward value = report_objective.
        # Backward gradient = gradient_objective.
        output = (
            gradient_objective
            + (
                report_objective.detach()
                - gradient_objective.detach()
            )
        )

        self.step_count += 1

        if (
            self.log_every > 0
            and self.step_count
            % self.log_every == 0
        ):
            conflict_rate = (
                conflict_count
                / max(pair_count, 1)
            )

            print(
                "[JOINT_PCSAFE]",
                {
                    "step": self.step_count,
                    "tasks": len(tasks),
                    "groups":
                        diagnostic_rows,
                    "conflict_count":
                        conflict_count,
                    "pair_count":
                        pair_count,
                    "conflict_rate":
                        conflict_rate,
                    "report_loss":
                        float(
                            report_objective
                            .detach()
                        ),
                    "anchor_loss":
                        float(
                            anchor_loss.detach()
                        ),
                },
                flush=True,
            )

        return output


class Exp_Joint_TGCA_PCSafe(
    Exp_Joint_TGCA
):
    def __init__(self, args) -> None:
        self._pcsafe_teacher = None
        self._pcsafe_teacher_output = None
        self._pcsafe_teacher_installed = False
        self._pcsafe_original_forward = None

        super().__init__(args)

    @staticmethod
    def _is_safety_head(
        name: str,
    ) -> bool:
        if name in SAFETY_HEAD_NAMES:
            return True

        return any(
            name.startswith(prefix)
            for prefix in SAFETY_HEAD_PREFIXES
        )

    def _joint_configure_trainable(
        self,
    ) -> None:
        core = self._joint_core_model()

        names = []
        trainable_count = 0
        total_count = 0

        for name, parameter in (
            core.named_parameters()
        ):
            total_count += parameter.numel()

            trainable = self._is_safety_head(
                name
            )

            parameter.requires_grad = trainable

            if trainable:
                names.append(name)
                trainable_count += (
                    parameter.numel()
                )

        if not names:
            available = [
                name
                for name, _
                in core.named_parameters()
            ]

            raise RuntimeError(
                "No safety-head parameters found.\n"
                + "\n".join(available)
            )

        self._joint_tail_names = set()

        print(
            "[JOINT_PCSAFE_TRAINABLE]",
            {
                "trainable":
                    trainable_count,
                "total":
                    total_count,
                "ratio":
                    trainable_count
                    / max(total_count, 1),
                "names":
                    names,
            },
            flush=True,
        )

    def _install_frozen_teacher(
        self,
    ) -> None:
        if self._pcsafe_teacher_installed:
            return

        core = self._joint_core_model()

        teacher = copy.deepcopy(core)

        teacher.to(self.device)
        teacher.eval()

        for parameter in teacher.parameters():
            parameter.requires_grad = False

        original_forward = core.forward

        self._pcsafe_teacher = teacher
        self._pcsafe_original_forward = (
            original_forward
        )

        def wrapped_forward(
            *args,
            **kwargs,
        ):
            if (
                core.training
                and torch.is_grad_enabled()
            ):
                with torch.no_grad():
                    self._pcsafe_teacher_output = (
                        teacher(
                            *args,
                            **kwargs,
                        )
                    )
            else:
                self._pcsafe_teacher_output = None

            return original_forward(
                *args,
                **kwargs,
            )

        core.forward = wrapped_forward

        self._pcsafe_teacher_installed = True

        print(
            "[JOINT_PCSAFE_TEACHER]",
            {
                "installed": True,
                "device": str(self.device),
            },
            flush=True,
        )

    def _select_criterion(self):
        return TemporalPCSafeLoss(self)

    def train(self, setting):
        self._joint_load_initial_checkpoint()
        self._install_frozen_teacher()

        # Parent train() sees the checkpoint as already
        # loaded, snapshots the trainable safety head,
        # and enters the normal training loop.
        return super().train(setting)
