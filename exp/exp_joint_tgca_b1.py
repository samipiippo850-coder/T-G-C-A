from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

try:
    from .exp_joint_tgca import Exp_Joint_TGCA as _Base
except ImportError:
    from .exp_joint_tgca_pcsafe import (
        Exp_Joint_TGCA_PCSafe as _Base,
    )


class DualMetricLoss(nn.Module):
    """Directly optimize the two reported metrics."""

    def __init__(
        self,
        mse_weight: float = 0.7,
        mae_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.mae_weight = float(mae_weight)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        mse = F.mse_loss(
            prediction,
            target,
        )

        mae = F.l1_loss(
            prediction,
            target,
        )

        return (
            self.mse_weight * mse
            + self.mae_weight * mae
        )



class RelaxationPINNLoss(nn.Module):
    """
    Target-free relaxation PINN-lite regularizer.

    Forecast objective:
        w_mse * MSE + w_mae * MAE

    Relaxation residual:
        dy_t + (y_t - trend_t) / tau

    The curvature of the estimated low-frequency trend
    is used only as a detached confidence gate.
    """

    def __init__(
        self,
        mse_weight: float = 0.7,
        mae_weight: float = 0.3,
        physics_weight: float = 1e-4,
        relaxation_tau: float = 72.0,
        trend_kernel: int = 25,
        charbonnier_eps: float = 1e-3,
    ) -> None:
        super().__init__()

        self.mse_weight = float(mse_weight)
        self.mae_weight = float(mae_weight)
        self.physics_weight = float(physics_weight)
        self.relaxation_tau = float(relaxation_tau)
        self.trend_kernel = int(trend_kernel)
        self.charbonnier_eps = float(charbonnier_eps)

        self._call_count = 0

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        mse = F.mse_loss(
            prediction,
            target,
        )

        mae = F.l1_loss(
            prediction,
            target,
        )

        forecast_loss = (
            self.mse_weight * mse
            + self.mae_weight * mae
        )

        if (
            self.physics_weight <= 0.0
            or prediction.ndim != 3
            or prediction.shape[1] < 5
        ):
            return forecast_loss

        horizon = int(prediction.shape[1])

        kernel = min(
            self.trend_kernel,
            horizon if horizon % 2 == 1 else horizon - 1,
        )

        kernel = max(3, kernel)

        if kernel % 2 == 0:
            kernel -= 1

        padding = kernel // 2

        # [B, T, C] -> [B, C, T]
        channel_first = prediction.transpose(1, 2)

        padded = F.pad(
            channel_first,
            (padding, padding),
            mode="replicate",
        )

        trend = F.avg_pool1d(
            padded,
            kernel_size=kernel,
            stride=1,
        ).transpose(1, 2)

        velocity = (
            prediction[:, 1:, :]
            - prediction[:, :-1, :]
        )

        displacement = (
            prediction[:, :-1, :]
            - trend[:, :-1, :]
        )

        relaxation_residual = (
            velocity
            + displacement / self.relaxation_tau
        )

        curvature = (
            trend[:, 2:, :]
            - 2.0 * trend[:, 1:-1, :]
            + trend[:, :-2, :]
        )

        # Do not allow the network to lower the loss
        # merely by manipulating the confidence gate.
        curvature_scale = (
            curvature.detach()
            .abs()
            .median()
            .clamp_min(1e-6)
        )

        confidence_gate = torch.exp(
            -curvature.detach().abs()
            / curvature_scale
        ).clamp(
            min=0.05,
            max=1.0,
        )

        # Align residual and curvature locations.
        relaxation_residual = (
            relaxation_residual[:, 1:, :]
        )

        velocity_scale = (
            velocity.detach()
            .abs()
            .mean()
            .clamp_min(1e-4)
        )

        normalized_residual = (
            relaxation_residual
            / velocity_scale
        )

        robust_residual = (
            torch.sqrt(
                normalized_residual.pow(2)
                + self.charbonnier_eps ** 2
            )
            - self.charbonnier_eps
        )

        physics_loss = (
            confidence_gate
            * robust_residual
        ).sum() / confidence_gate.sum().clamp_min(1.0)

        total_loss = (
            forecast_loss
            + self.physics_weight * physics_loss
        )

        self._call_count += 1

        if self._call_count % 500 == 0:
            print(
                "[P1_PINN_LOSS]",
                {
                    "forecast":
                        float(
                            forecast_loss.detach().cpu()
                        ),
                    "mse":
                        float(mse.detach().cpu()),
                    "mae":
                        float(mae.detach().cpu()),
                    "physics":
                        float(
                            physics_loss.detach().cpu()
                        ),
                    "physics_weight":
                        self.physics_weight,
                    "tau":
                        self.relaxation_tau,
                    "trend_kernel":
                        kernel,
                    "gate_mean":
                        float(
                            confidence_gate.mean()
                            .detach().cpu()
                        ),
                },
                flush=True,
            )

        return total_loss



class MultiScaleRelaxationPINNLoss(nn.Module):
    """
    Target-free multi-timescale relaxation regularizer.

    Timescale weights are determined from detached forecast
    velocity and trend curvature, so the model cannot lower
    the physics loss merely by manipulating the router.
    """

    def __init__(
        self,
        mse_weight: float = 0.6,
        mae_weight: float = 0.4,
        physics_weight: float = 1e-4,
        tau_scales=(48.0, 72.0, 120.0),
        trend_kernel: int = 25,
        temperature: float = 0.75,
        charbonnier_eps: float = 1e-3,
    ) -> None:
        super().__init__()

        if len(tau_scales) != 3:
            raise ValueError(
                "P2 requires exactly three tau scales"
            )

        self.mse_weight = float(mse_weight)
        self.mae_weight = float(mae_weight)
        self.physics_weight = float(physics_weight)
        self.tau_scales = tuple(
            float(value)
            for value in tau_scales
        )
        self.trend_kernel = int(trend_kernel)
        self.temperature = max(
            float(temperature),
            1e-3,
        )
        self.charbonnier_eps = float(
            charbonnier_eps
        )

        self._call_count = 0

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        mse = F.mse_loss(
            prediction,
            target,
        )
        mae = F.l1_loss(
            prediction,
            target,
        )

        forecast_loss = (
            self.mse_weight * mse
            + self.mae_weight * mae
        )

        if (
            self.physics_weight <= 0.0
            or prediction.ndim != 3
            or prediction.shape[1] < 5
        ):
            return forecast_loss

        horizon = int(prediction.shape[1])

        kernel = min(
            self.trend_kernel,
            horizon
            if horizon % 2 == 1
            else horizon - 1,
        )
        kernel = max(3, kernel)

        if kernel % 2 == 0:
            kernel -= 1

        padding = kernel // 2

        # [B, T, C] -> [B, C, T]
        channel_first = prediction.transpose(
            1,
            2,
        )

        padded = F.pad(
            channel_first,
            (padding, padding),
            mode="replicate",
        )

        trend = F.avg_pool1d(
            padded,
            kernel_size=kernel,
            stride=1,
        ).transpose(
            1,
            2,
        )

        velocity = (
            prediction[:, 1:, :]
            - prediction[:, :-1, :]
        )

        displacement = (
            prediction[:, :-1, :]
            - trend[:, :-1, :]
        )

        curvature = (
            trend[:, 2:, :]
            - 2.0 * trend[:, 1:-1, :]
            + trend[:, :-2, :]
        )

        # Align the residual with curvature positions.
        velocity_aligned = velocity[:, 1:, :]
        displacement_aligned = displacement[:, 1:, :]

        residuals = []

        for tau in self.tau_scales:
            residuals.append(
                velocity_aligned
                + displacement_aligned / tau
            )

        # [B, T-2, C, 3]
        residual_stack = torch.stack(
            residuals,
            dim=-1,
        )

        detached_velocity = (
            velocity_aligned.detach().abs()
        )
        detached_curvature = (
            curvature.detach().abs()
        )

        velocity_reference = (
            detached_velocity.median()
            .clamp_min(1e-6)
        )
        curvature_reference = (
            detached_curvature.median()
            .clamp_min(1e-6)
        )

        # Per-sample/per-channel dynamical activity.
        velocity_activity = (
            detached_velocity.mean(
                dim=1,
                keepdim=True,
            )
            / velocity_reference
        )

        curvature_activity = (
            detached_curvature.mean(
                dim=1,
                keepdim=True,
            )
            / curvature_reference
        )

        # Fast state: strong velocity/curvature.
        fast_logit = (
            0.9 * velocity_activity
            + 0.6 * curvature_activity
        )

        # Intermediate state: activity near the
        # historical normalized scale of one.
        medium_logit = (
            -torch.abs(
                velocity_activity - 1.0
            )
            - 0.5
            * torch.abs(
                curvature_activity - 1.0
            )
        )

        # Slow state: smooth, low-activity regime.
        slow_logit = (
            -0.8 * velocity_activity
            - 0.3 * curvature_activity
        )

        # [B, 1, C, 3]
        tau_logits = torch.stack(
            [
                fast_logit,
                medium_logit,
                slow_logit,
            ],
            dim=-1,
        )

        tau_weights = torch.softmax(
            tau_logits / self.temperature,
            dim=-1,
        ).detach()

        # Local normalization prevents high-amplitude
        # channels from dominating the physics loss.
        local_velocity_scale = (
            detached_velocity.mean(
                dim=1,
                keepdim=True,
            )
            .clamp_min(1e-4)
            .unsqueeze(-1)
        )

        normalized_residual = (
            residual_stack
            / local_velocity_scale
        )

        robust_residual = (
            torch.sqrt(
                normalized_residual.pow(2)
                + self.charbonnier_eps ** 2
            )
            - self.charbonnier_eps
        )

        mixed_residual = (
            robust_residual
            * tau_weights
        ).sum(
            dim=-1
        )

        # Physics confidence is reduced in sharply
        # changing curvature regimes.
        confidence_gate = torch.exp(
            -detached_curvature
            / curvature_reference
        ).clamp(
            min=0.05,
            max=1.0,
        )

        physics_loss = (
            confidence_gate
            * mixed_residual
        ).sum() / confidence_gate.sum().clamp_min(
            1.0
        )

        total_loss = (
            forecast_loss
            + self.physics_weight
            * physics_loss
        )

        self._call_count += 1

        if self._call_count % 500 == 0:
            mean_tau_weights = (
                tau_weights.mean(
                    dim=(0, 1, 2)
                )
                .detach()
                .cpu()
                .tolist()
            )

            print(
                "[P2_MULTISCALE_PINN_LOSS]",
                {
                    "forecast":
                        float(
                            forecast_loss
                            .detach().cpu()
                        ),
                    "mse":
                        float(mse.detach().cpu()),
                    "mae":
                        float(mae.detach().cpu()),
                    "physics":
                        float(
                            physics_loss
                            .detach().cpu()
                        ),
                    "physics_weight":
                        self.physics_weight,
                    "tau_scales":
                        self.tau_scales,
                    "tau_weight_mean":
                        mean_tau_weights,
                    "gate_mean":
                        float(
                            confidence_gate.mean()
                            .detach().cpu()
                        ),
                    "temperature":
                        self.temperature,
                },
                flush=True,
            )

        return total_loss


class Exp_Joint_TGCA_B1(_Base):
    """
    B1 screening:
      - TGCA adapter trainable
      - prediction/output modules trainable
      - final backbone leaf modules trainable
      - differential learning rates
    """

    def __init__(self, args):
        # Force the actual experiment configuration before
        # the model, loss and optimizer are constructed.
        args.alpha = float(
            os.environ.get(
                "B1_FORCE_ALPHA",
                "0.0",
            )
        )
        args.rec_lambda = float(
            os.environ.get(
                "B1_FORCE_REC_LAMBDA",
                "1.0",
            )
        )
        args.auxi_loss = os.environ.get(
            "B1_FORCE_AUXI_LOSS",
            "MAE",
        )
        args.module_first = int(
            os.environ.get(
                "B1_FORCE_MODULE_FIRST",
                "1",
            )
        )
        args.distil = True

        print(
            "[B1_CANONICAL_ARGS]",
            {
                "alpha": args.alpha,
                "rec_lambda": args.rec_lambda,
                "auxi_loss": args.auxi_loss,
                "module_first": args.module_first,
                "distil": args.distil,
            },
            flush=True,
        )

        # Disable every inherited auxiliary objective.
        args.auxi_lambda = 0.0
        args.reg_lambda = 0.0
        args.decorr_lambda = 0.0
        args.hf_lambda = 0.0
        args.use_hf_loss = 0
        args.use_event_weight = 0
        args.use_quantile_loss = 0
        args.use_turn_loss = 0

        os.environ["B1_CLEAN_FORECAST_LOSS"] = (
            os.environ.get(
                "B1_CLEAN_FORECAST_LOSS",
                "1",
            )
        )

        print(
            "[B1_CLEAN_LOSS_ARGS]",
            {
                "clean_forecast_loss":
                    os.environ[
                        "B1_CLEAN_FORECAST_LOSS"
                    ],
                "auxi_lambda":
                    args.auxi_lambda,
                "reg_lambda":
                    args.reg_lambda,
                "decorr_lambda":
                    args.decorr_lambda,
                "hf_lambda":
                    args.hf_lambda,
            },
            flush=True,
        )

        super().__init__(args)
        self._configure_b1_trainables()

    def _unwrap_model(self):
        if hasattr(self.model, "module"):
            return self.model.module

        return self.model

    @staticmethod
    def _is_adapter_parameter(
        name: str,
    ) -> bool:
        lowered = name.lower()

        return (
            lowered.startswith("adapter.")
            or ".adapter." in lowered
        )

    def _configure_b1_trainables(self) -> None:
        model = self._unwrap_model()

        last_leaf_count = int(
            os.environ.get(
                "B1_LAST_LEAF_MODULES",
                "8",
            )
        )

        maximum_backbone_parameters = int(
            os.environ.get(
                "B1_MAX_BACKBONE_PARAMS",
                "150000",
            )
        )

        # Begin from a known frozen state.
        for parameter in model.parameters():
            parameter.requires_grad = False

        named_parameters = dict(
            model.named_parameters()
        )

        adapter_names = {
            name
            for name in named_parameters
            if self._is_adapter_parameter(
                name
            )
        }

        # Direct parameters of leaf-like modules.
        leaf_modules = []

        for module_name, module in (
            model.named_modules()
        ):
            direct_parameters = list(
                module.named_parameters(
                    recurse=False
                )
            )

            if not direct_parameters:
                continue

            full_names = []

            for parameter_name, _ in (
                direct_parameters
            ):
                if module_name:
                    full_name = (
                        f"{module_name}."
                        f"{parameter_name}"
                    )
                else:
                    full_name = parameter_name

                if full_name in named_parameters:
                    full_names.append(
                        full_name
                    )

            full_names = [
                name
                for name in full_names
                if not self._is_adapter_parameter(
                    name
                )
            ]

            if full_names:
                leaf_modules.append(
                    (
                        module_name,
                        full_names,
                    )
                )

        selected_backbone_names = set()

        # Coherent module-prefix unfreezing for B3.
        prefix_specification = os.environ.get(
            "B1_BACKBONE_PREFIXES",
            "",
        ).strip()

        if prefix_specification:
            prefixes = tuple(
                prefix.strip()
                for prefix in
                prefix_specification.split(",")
                if prefix.strip()
            )

            selected_backbone_names.update(
                name
                for name in named_parameters
                if (
                    name.startswith(prefixes)
                    and not self._is_adapter_parameter(name)
                )
            )

            print(
                "[B1_PREFIX_UNFREEZE]",
                {
                    "prefixes": prefixes,
                    "parameter_tensors":
                        len(selected_backbone_names),
                    "parameters":
                        sum(
                            named_parameters[name].numel()
                            for name
                            in selected_backbone_names
                        ),
                },
                flush=True,
            )

        output_keywords = (
            "projection",
            "predict",
            "prediction",
            "forecast",
            "output",
            "head",
            "out_layer",
        )

        # Always include explicit output heads.
        for module_name, names in leaf_modules:
            lowered = module_name.lower()

            if any(
                keyword in lowered
                for keyword in output_keywords
            ):
                selected_backbone_names.update(
                    names
                )

        # Then include the last leaf modules, subject
        # to a parameter budget.
        selected_parameter_count = sum(
            named_parameters[name].numel()
            for name in selected_backbone_names
        )

        added_leaf_modules = 0

        for module_name, names in reversed(
            leaf_modules
        ):
            if added_leaf_modules >= last_leaf_count:
                break

            new_names = [
                name
                for name in names
                if name
                not in selected_backbone_names
            ]

            new_parameter_count = sum(
                named_parameters[name].numel()
                for name in new_names
            )

            if (
                selected_parameter_count
                + new_parameter_count
                > maximum_backbone_parameters
                and selected_backbone_names
            ):
                continue

            selected_backbone_names.update(
                new_names
            )

            selected_parameter_count += (
                new_parameter_count
            )

            added_leaf_modules += 1

        if not selected_backbone_names:
            raise RuntimeError(
                "B1 failed to identify any "
                "backbone parameters."
            )

        for name in adapter_names:
            named_parameters[
                name
            ].requires_grad = True

        for name in selected_backbone_names:
            named_parameters[
                name
            ].requires_grad = True

        self._b1_adapter_names = sorted(
            adapter_names
        )

        self._b1_backbone_names = sorted(
            selected_backbone_names
        )

        trainable = [
            (
                name,
                parameter.numel(),
            )
            for name, parameter
            in model.named_parameters()
            if parameter.requires_grad
        ]

        print(
            "[B1_TRAINABLE]",
            {
                "base_class":
                    type(self).__mro__[1].__name__,
                "adapter_tensors":
                    len(
                        self._b1_adapter_names
                    ),
                "adapter_parameters":
                    sum(
                        named_parameters[
                            name
                        ].numel()
                        for name
                        in self._b1_adapter_names
                    ),
                "backbone_tensors":
                    len(
                        self._b1_backbone_names
                    ),
                "backbone_parameters":
                    sum(
                        named_parameters[
                            name
                        ].numel()
                        for name
                        in self._b1_backbone_names
                    ),
                "total_trainable":
                    sum(
                        count
                        for _, count in trainable
                    ),
                "backbone_names":
                    self._b1_backbone_names,
            },
            flush=True,
        )

    def _select_criterion(self):
        mse_weight = float(
            os.environ.get(
                "B1_MSE_WEIGHT",
                "0.7",
            )
        )

        mae_weight = float(
            os.environ.get(
                "B1_MAE_WEIGHT",
                "0.3",
            )
        )

        if (
            os.environ.get(
                "B1_PINN_MULTISCALE",
                "0",
            ) == "1"
        ):
            tau_scales = tuple(
                float(value.strip())
                for value in os.environ.get(
                    "P2_TAU_SCALES",
                    "48,72,120",
                ).split(",")
                if value.strip()
            )

            criterion = (
                MultiScaleRelaxationPINNLoss(
                    mse_weight=mse_weight,
                    mae_weight=mae_weight,
                    physics_weight=float(
                        os.environ.get(
                            "P2_PHYSICS_WEIGHT",
                            "1e-4",
                        )
                    ),
                    tau_scales=tau_scales,
                    trend_kernel=int(
                        os.environ.get(
                            "P2_TREND_KERNEL",
                            "25",
                        )
                    ),
                    temperature=float(
                        os.environ.get(
                            "P2_TEMPERATURE",
                            "0.75",
                        )
                    ),
                )
            )

            print(
                "[P2_MULTISCALE_PINN_CRITERION]",
                {
                    "mse_weight":
                        mse_weight,
                    "mae_weight":
                        mae_weight,
                    "physics_weight":
                        criterion.physics_weight,
                    "tau_scales":
                        criterion.tau_scales,
                    "trend_kernel":
                        criterion.trend_kernel,
                    "temperature":
                        criterion.temperature,
                },
                flush=True,
            )

            return criterion

        if (
            os.environ.get(
                "B1_PINN_LITE",
                "0",
            ) == "1"
        ):
            criterion = RelaxationPINNLoss(
                mse_weight=mse_weight,
                mae_weight=mae_weight,
                physics_weight=float(
                    os.environ.get(
                        "P1_PHYSICS_WEIGHT",
                        "1e-4",
                    )
                ),
                relaxation_tau=float(
                    os.environ.get(
                        "P1_RELAXATION_TAU",
                        "72.0",
                    )
                ),
                trend_kernel=int(
                    os.environ.get(
                        "P1_TREND_KERNEL",
                        "25",
                    )
                ),
            )

            print(
                "[P1_PINN_CRITERION]",
                {
                    "mse_weight":
                        mse_weight,
                    "mae_weight":
                        mae_weight,
                    "physics_weight":
                        criterion.physics_weight,
                    "relaxation_tau":
                        criterion.relaxation_tau,
                    "trend_kernel":
                        criterion.trend_kernel,
                },
                flush=True,
            )

            return criterion

        return DualMetricLoss(
            mse_weight=mse_weight,
            mae_weight=mae_weight,
        )

    def _select_optimizer(self):
        model = self._unwrap_model()

        adapter_parameters = []
        backbone_parameters = []

        for name, parameter in (
            model.named_parameters()
        ):
            if not parameter.requires_grad:
                continue

            if self._is_adapter_parameter(
                name
            ):
                adapter_parameters.append(
                    parameter
                )
            else:
                backbone_parameters.append(
                    parameter
                )

        if not adapter_parameters:
            raise RuntimeError(
                "No trainable adapter parameters."
            )

        if not backbone_parameters:
            raise RuntimeError(
                "No trainable backbone parameters."
            )

        module_lr = float(
            os.environ.get(
                "B1_MODULE_LR",
                "2e-5",
            )
        )

        backbone_lr = float(
            os.environ.get(
                "B1_BACKBONE_LR",
                "5e-6",
            )
        )

        weight_decay = float(
            os.environ.get(
                "B1_WEIGHT_DECAY",
                "1e-4",
            )
        )

        optimizer = optim.AdamW(
            [
                {
                    "params":
                        adapter_parameters,
                    "lr":
                        module_lr,
                    "weight_decay":
                        weight_decay,
                    "group_name":
                        "tgca",
                },
                {
                    "params":
                        backbone_parameters,
                    "lr":
                        backbone_lr,
                    "weight_decay":
                        weight_decay,
                    "group_name":
                        "backbone",
                },
            ],
            lr=module_lr,
        )

        print(
            "[B1_OPTIMIZER_GROUPS]",
            {
                "tgca": {
                    "lr": module_lr,
                    "tensors":
                        len(
                            adapter_parameters
                        ),
                    "parameters":
                        sum(
                            parameter.numel()
                            for parameter
                            in adapter_parameters
                        ),
                },
                "backbone": {
                    "lr": backbone_lr,
                    "tensors":
                        len(
                            backbone_parameters
                        ),
                    "parameters":
                        sum(
                            parameter.numel()
                            for parameter
                            in backbone_parameters
                        ),
                },
                "weight_decay":
                    weight_decay,
            },
            flush=True,
        )

        return optimizer
