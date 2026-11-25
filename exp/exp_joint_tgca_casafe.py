from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import minimize

from exp.exp_joint_tgca_pcsafe import (
    Exp_Joint_TGCA_PCSafe,
    TemporalPCSafeLoss,
)


class TemporalCASafeLoss(TemporalPCSafeLoss):
    """
    CAGrad replacement for the PCGrad gradient merger.

    The task construction, frozen teacher, gain penalties,
    direction penalties and trainable parameters remain
    identical to A3.
    """

    def __init__(self, experiment) -> None:
        super().__init__(experiment)

        self.cagrad_c = float(
            os.environ.get(
                "JOINT_CAGRAD_C",
                "0.4",
            )
        )

        if not 0.0 <= self.cagrad_c < 1.0:
            raise ValueError(
                "JOINT_CAGRAD_C must satisfy 0 <= c < 1"
            )

        self.cagrad_maxiter = int(
            os.environ.get(
                "JOINT_CAGRAD_MAXITER",
                "100",
            )
        )

        self.cagrad_tol = float(
            os.environ.get(
                "JOINT_CAGRAD_TOL",
                "1e-8",
            )
        )

        self.casafe_log_every = int(
            os.environ.get(
                "JOINT_CAGRAD_LOG_EVERY",
                "100",
            )
        )

        # Disable inherited [JOINT_PCSAFE] logging.
        self.log_every = 0

    @staticmethod
    def _flatten_task(
        gradient: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        return torch.cat(
            [
                value.reshape(-1)
                for value in gradient
            ],
            dim=0,
        )

    @staticmethod
    def _unflatten(
        vector: torch.Tensor,
        reference: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        result = []
        offset = 0

        for tensor in reference:
            count = tensor.numel()

            result.append(
                vector[offset:offset + count]
                .reshape_as(tensor)
            )

            offset += count

        if offset != vector.numel():
            raise RuntimeError(
                "CAGrad unflatten size mismatch"
            )

        return result

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
        """
        Override the parent merge method with CAGrad.

        The method name is kept because the parent's forward()
        calls _pcgrad_merge().
        """
        flat = torch.stack(
            [
                self._flatten_task(task)
                for task in task_gradients
            ],
            dim=1,
        )

        # flat shape: [parameter_count, task_count]
        task_count = flat.shape[1]

        average_gradient = torch.mean(
            flat,
            dim=1,
        )

        average_norm = torch.linalg.vector_norm(
            average_gradient
        )

        if (
            not torch.isfinite(average_norm)
            or float(average_norm.detach()) <= self.pc_eps
        ):
            merged = self._unflatten(
                average_gradient,
                task_gradients[0],
            )

            return merged, 0, task_count

        gram_tensor = (
            flat.transpose(0, 1)
            @ flat
        ).double()

        gram = (
            gram_tensor
            .detach()
            .cpu()
            .numpy()
        )

        initial = np.full(
            task_count,
            1.0 / task_count,
            dtype=np.float64,
        )

        average_weights = initial.copy()

        radius = (
            self.cagrad_c
            * float(average_norm.detach().cpu())
        )

        eps = self.pc_eps

        def objective(weights: np.ndarray) -> float:
            weights = np.asarray(
                weights,
                dtype=np.float64,
            )

            mixed_norm_sq = float(
                weights @ gram @ weights
            )

            alignment = float(
                weights @ gram @ average_weights
            )

            return (
                alignment
                + radius
                * np.sqrt(
                    max(mixed_norm_sq, 0.0)
                    + eps
                )
            )

        def jacobian(weights: np.ndarray) -> np.ndarray:
            weights = np.asarray(
                weights,
                dtype=np.float64,
            )

            mixed_norm_sq = float(
                weights @ gram @ weights
            )

            denominator = np.sqrt(
                max(mixed_norm_sq, 0.0)
                + eps
            )

            return (
                gram @ average_weights
                + radius
                * (gram @ weights)
                / denominator
            )

        constraints = {
            "type": "eq",
            "fun": lambda weights:
                float(np.sum(weights) - 1.0),
            "jac": lambda weights:
                np.ones_like(
                    weights,
                    dtype=np.float64,
                ),
        }

        result = minimize(
            objective,
            initial,
            jac=jacobian,
            bounds=[
                (0.0, 1.0)
                for _ in range(task_count)
            ],
            constraints=constraints,
            method="SLSQP",
            options={
                "maxiter":
                    self.cagrad_maxiter,
                "ftol":
                    self.cagrad_tol,
                "disp":
                    False,
            },
        )

        if (
            not result.success
            or not np.all(
                np.isfinite(result.x)
            )
        ):
            weights_np = initial
            solver_status = (
                "FALLBACK_UNIFORM:"
                + str(result.message)
            )
        else:
            weights_np = result.x
            solver_status = "SOLVED"

        weights = torch.as_tensor(
            weights_np,
            device=flat.device,
            dtype=flat.dtype,
        )

        mixed_gradient = flat @ weights
        mixed_norm = torch.linalg.vector_norm(
            mixed_gradient
        )

        scale = (
            radius
            / (
                float(
                    mixed_norm.detach()
                )
                + self.pc_eps
            )
        )

        merged_flat = (
            average_gradient
            + scale * mixed_gradient
        )

        alignments = (
            flat.transpose(0, 1)
            @ merged_flat
        )

        unsafe_task_count = int(
            torch.sum(
                alignments < 0.0
            ).item()
        )

        next_step = self.step_count + 1

        if (
            self.casafe_log_every > 0
            and next_step
            % self.casafe_log_every == 0
        ):
            print(
                "[JOINT_CASAFE]",
                {
                    "step": next_step,
                    "c": self.cagrad_c,
                    "solver_status":
                        solver_status,
                    "weights":
                        weights_np.tolist(),
                    "unsafe_task_count":
                        unsafe_task_count,
                    "min_task_alignment":
                        float(
                            alignments.min()
                            .detach()
                        ),
                    "mean_task_alignment":
                        float(
                            alignments.mean()
                            .detach()
                        ),
                    "average_gradient_norm":
                        float(
                            average_norm.detach()
                        ),
                    "merged_gradient_norm":
                        float(
                            torch.linalg.vector_norm(
                                merged_flat
                            ).detach()
                        ),
                },
                flush=True,
            )

        merged = self._unflatten(
            merged_flat,
            task_gradients[0],
        )

        # Parent expects two diagnostic counts.
        return (
            merged,
            unsafe_task_count,
            task_count,
        )


class Exp_Joint_TGCA_CASafe(
    Exp_Joint_TGCA_PCSafe
):
    def _select_criterion(self):
        return TemporalCASafeLoss(self)
