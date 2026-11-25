from __future__ import annotations

import os
import weakref
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(
        name,
        "1" if default else "0",
    ).strip().lower()

    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


class PatchStructuralLoss(nn.Module):
    """
    Architecture-independent implementation of Patch-wise
    Structural Loss.

    Inputs:
        prediction, target: [B, H, C]

    Components:
        - local linear-correlation loss
        - local distribution/variance KL loss
        - local patch-mean loss

    Patch length is estimated from the dominant frequency of
    the training target, following the official PS Loss logic.
    """

    def __init__(
        self,
        patch_max: int = 24,
        patch_min: int = 4,
        dynamic_weighting: bool = True,
        weight_min: float = 0.10,
        weight_max: float = 10.0,
        eps: float = 1e-5,
    ):
        super().__init__()

        if patch_max < 2:
            raise ValueError(
                f"patch_max must be >= 2, got {patch_max}"
            )

        self.patch_max = int(patch_max)
        self.patch_min = int(patch_min)
        self.dynamic_weighting = bool(
            dynamic_weighting
        )
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.eps = float(eps)

    def _adaptive_patch_length(
        self,
        target: torch.Tensor,
    ) -> Tuple[int, int, int]:
        horizon = int(target.shape[1])

        if horizon < 2:
            return 1, 1, 0

        with torch.no_grad():
            spectrum = torch.fft.rfft(
                target.detach().float(),
                dim=1,
            ).abs()

            # Average over batch and channels.
            frequency_strength = spectrum.mean(
                dim=(0, 2)
            )

            if frequency_strength.numel() > 0:
                frequency_strength[0] = 0.0

            dominant_index = int(
                torch.argmax(
                    frequency_strength
                ).item()
            )

        if dominant_index <= 0:
            period = horizon
        else:
            period = max(
                2,
                horizon // dominant_index,
            )

        patch_len = max(
            2,
            period // 2,
        )

        patch_len = min(
            patch_len,
            self.patch_max,
            horizon,
        )

        if horizon >= self.patch_min:
            patch_len = max(
                patch_len,
                self.patch_min,
            )

        patch_len = min(
            patch_len,
            horizon,
        )

        stride = max(
            1,
            patch_len // 2,
        )

        return (
            int(patch_len),
            int(stride),
            dominant_index,
        )

    @staticmethod
    def _create_patches(
        value: torch.Tensor,
        patch_len: int,
        stride: int,
    ) -> torch.Tensor:
        """
        [B, H, C] -> [B, C, N, P]
        """
        if value.ndim != 3:
            raise ValueError(
                f"Expected [B,H,C], got {value.shape}"
            )

        channel_first = value.permute(
            0,
            2,
            1,
        )

        return channel_first.unfold(
            dimension=2,
            size=patch_len,
            step=stride,
        )

    def _components(
        self,
        target_patch: torch.Tensor,
        prediction_patch: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        target_mean = target_patch.mean(
            dim=-1,
            keepdim=True,
        )

        prediction_mean = prediction_patch.mean(
            dim=-1,
            keepdim=True,
        )

        target_centered = (
            target_patch - target_mean
        )

        prediction_centered = (
            prediction_patch
            - prediction_mean
        )

        target_var = (
            target_centered.pow(2)
            .mean(
                dim=-1,
                keepdim=True,
            )
        )

        prediction_var = (
            prediction_centered.pow(2)
            .mean(
                dim=-1,
                keepdim=True,
            )
        )

        target_std = torch.sqrt(
            target_var + self.eps
        )

        prediction_std = torch.sqrt(
            prediction_var + self.eps
        )

        covariance = (
            target_centered
            * prediction_centered
        ).mean(
            dim=-1,
            keepdim=True,
        )

        correlation = covariance / (
            target_std
            * prediction_std
            + self.eps
        )

        correlation = correlation.clamp(
            -1.0,
            1.0,
        )

        correlation_loss = (
            1.0 - correlation
        ).mean()

        # Official implementation represents local shape /
        # variance through KL divergence over the patch.
        target_probability = F.softmax(
            target_patch,
            dim=-1,
        )

        prediction_log_probability = (
            F.log_softmax(
                prediction_patch,
                dim=-1,
            )
        )

        variance_loss = F.kl_div(
            prediction_log_probability,
            target_probability,
            reduction="none",
        ).sum(
            dim=-1
        ).mean()

        mean_loss = (
            target_mean
            - prediction_mean
        ).abs().mean()

        return (
            correlation_loss,
            variance_loss,
            mean_loss,
        )

    def _gradient_weights(
        self,
        prediction: torch.Tensor,
        components: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        if (
            not self.dynamic_weighting
            or not torch.is_grad_enabled()
            or not prediction.requires_grad
        ):
            return prediction.new_ones(3)

        norms = []

        for component in components:
            gradient = torch.autograd.grad(
                outputs=component,
                inputs=prediction,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]

            if gradient is None:
                norm = prediction.new_tensor(1.0)
            else:
                norm = torch.sqrt(
                    gradient.detach()
                    .pow(2)
                    .mean()
                    + self.eps
                )

            norms.append(norm)

        norm_tensor = torch.stack(norms)

        average_norm = norm_tensor.mean()

        weights = average_norm / (
            norm_tensor + self.eps
        )

        weights = weights.clamp(
            self.weight_min,
            self.weight_max,
        )

        # Preserve the overall PS-loss scale.
        weights = weights / (
            weights.mean() + self.eps
        )

        return weights.detach()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if prediction.shape != target.shape:
            raise ValueError(
                "PS loss shape mismatch: "
                f"{prediction.shape} vs {target.shape}"
            )

        if prediction.ndim != 3:
            raise ValueError(
                "PS loss requires [B,H,C], "
                f"got {prediction.shape}"
            )

        patch_len, stride, dominant_index = (
            self._adaptive_patch_length(
                target
            )
        )

        target_patch = self._create_patches(
            target,
            patch_len,
            stride,
        )

        prediction_patch = self._create_patches(
            prediction,
            patch_len,
            stride,
        )

        components = self._components(
            target_patch,
            prediction_patch,
        )

        weights = self._gradient_weights(
            prediction,
            components,
        )

        correlation_loss, variance_loss, mean_loss = (
            components
        )

        total = (
            weights[0] * correlation_loss
            + weights[1] * variance_loss
            + weights[2] * mean_loss
        )

        logs = {
            "ps_total":
                float(total.detach().cpu()),
            "ps_corr":
                float(
                    correlation_loss.detach().cpu()
                ),
            "ps_var":
                float(
                    variance_loss.detach().cpu()
                ),
            "ps_mean":
                float(
                    mean_loss.detach().cpu()
                ),
            "ps_w_corr":
                float(weights[0].cpu()),
            "ps_w_var":
                float(weights[1].cpu()),
            "ps_w_mean":
                float(weights[2].cpu()),
            "ps_patch_len":
                int(patch_len),
            "ps_stride":
                int(stride),
            "ps_dominant_index":
                int(dominant_index),
        }

        return total, logs


class TGCACompositePSLoss(nn.Module):
    """
    Training:
        point_loss + ps_lambda * PS_loss

    Validation/test:
        point_loss only

    Validation therefore remains directly comparable to the
    original MSE-based early-stopping criterion.
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        model: nn.Module,
        ps_lambda: float,
        patch_max: int,
        dynamic_weighting: bool,
    ):
        super().__init__()

        self.base_criterion = base_criterion

        # Do not register the forecasting model as a child of
        # the criterion.
        object.__setattr__(
            self,
            "_model_ref",
            weakref.proxy(model),
        )

        self.ps_lambda = float(ps_lambda)

        self.ps_loss = PatchStructuralLoss(
            patch_max=patch_max,
            dynamic_weighting=dynamic_weighting,
        )

        self._print_count = 0

        print(
            "[TGCA_PS_INIT]",
            {
                "ps_lambda": self.ps_lambda,
                "patch_max": patch_max,
                "dynamic_weighting":
                    dynamic_weighting,
                "base_criterion":
                    type(base_criterion).__name__,
            },
            flush=True,
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        point_loss = self.base_criterion(
            prediction,
            target,
        )

        model = object.__getattribute__(
            self,
            "_model_ref",
        )

        # Validation and test must remain point-wise MSE.
        if (
            not model.training
            or not torch.is_grad_enabled()
            or self.ps_lambda <= 0.0
        ):
            return point_loss

        structural_loss, logs = self.ps_loss(
            prediction,
            target,
        )

        # Optional diagnostic: compare the weighted PS
        # gradient with the point-loss gradient at the model
        # output. This does not alter the eventual backward
        # pass because the graph is retained.
        grad_audit_enabled = (
            os.environ.get(
                "TGCA_PS_GRAD_AUDIT",
                "0",
            ) == "1"
        )

        grad_audit_count = getattr(
            self,
            "_grad_audit_count",
            0,
        )

        if (
            grad_audit_enabled
            and grad_audit_count < 5
            and torch.is_grad_enabled()
            and prediction.requires_grad
        ):
            point_gradient = torch.autograd.grad(
                point_loss,
                prediction,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]

            ps_gradient = torch.autograd.grad(
                structural_loss,
                prediction,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]

            if (
                point_gradient is not None
                and ps_gradient is not None
            ):
                point_gradient_rms = torch.sqrt(
                    point_gradient.detach()
                    .pow(2)
                    .mean()
                    + 1e-20
                )

                ps_gradient_rms = torch.sqrt(
                    ps_gradient.detach()
                    .pow(2)
                    .mean()
                    + 1e-20
                )

                weighted_ps_gradient_rms = (
                    self.ps_lambda
                    * ps_gradient_rms
                )

                gradient_ratio = (
                    weighted_ps_gradient_rms
                    / (
                        point_gradient_rms
                        + 1e-20
                    )
                )

                print(
                    "[TGCA_PS_GRAD_RATIO]",
                    {
                        "ps_lambda":
                            self.ps_lambda,
                        "point_gradient_rms":
                            float(
                                point_gradient_rms.cpu()
                            ),
                        "raw_ps_gradient_rms":
                            float(
                                ps_gradient_rms.cpu()
                            ),
                        "weighted_ps_gradient_rms":
                            float(
                                weighted_ps_gradient_rms.cpu()
                            ),
                        "weighted_ps_to_point_ratio":
                            float(
                                gradient_ratio.cpu()
                            ),
                    },
                    flush=True,
                )

            self._grad_audit_count = (
                grad_audit_count + 1
            )

        total_loss = (
            point_loss
            + self.ps_lambda
            * structural_loss
        )

        if self._print_count < 5:
            print(
                "[TGCA_PS_FORWARD]",
                {
                    "point_loss":
                        float(
                            point_loss.detach().cpu()
                        ),
                    **logs,
                    "ps_lambda":
                        self.ps_lambda,
                    "total_loss":
                        float(
                            total_loss.detach().cpu()
                        ),
                },
                flush=True,
            )

            self._print_count += 1

        return total_loss
