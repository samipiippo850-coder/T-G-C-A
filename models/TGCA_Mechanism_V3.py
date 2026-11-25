from __future__ import annotations

import copy
import math
import os
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import TGCA_Forecaster


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
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


class RiskConditionalMechanismAdapter(nn.Module):
    """
    Mechanism-driven residual field.

    Mechanisms:
      1. Tail–Energy:
         increments, intermittency and energy redistribution.

      2. Phase–Horizon:
         frequency structure and early/late phase drift.

    The residual field has three separately learned parts:

      direction(z, h, c) in [-1, 1]
      risk_cap(z, h, c) in [cap_min, cap_max]
      mechanism_mix(z, c)

    Residual energy is capped per sample × channel × horizon
    block rather than by one global sample RMS.
    """

    def __init__(self, configs) -> None:
        super().__init__()

        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.c_out)

        self.direction_blocks = env_int(
            "TGCA_V3_DIRECTION_BLOCKS",
            24,
        )

        if (
            self.direction_blocks <= 0
            or self.pred_len
            % self.direction_blocks
            != 0
        ):
            raise ValueError({
                "pred_len": self.pred_len,
                "direction_blocks":
                    self.direction_blocks,
                "message":
                    "pred_len must be divisible "
                    "by direction_blocks",
            })

        self.block_size = (
            self.pred_len
            // self.direction_blocks
        )

        self.patch_tokens = min(
            env_int(
                "TGCA_V3_PATCH_TOKENS",
                24,
            ),
            self.seq_len,
        )

        self.freq_bins = min(
            env_int(
                "TGCA_V3_FREQ_BINS",
                49,
            ),
            self.seq_len // 2 + 1,
        )

        self.hidden = env_int(
            "TGCA_V3_HIDDEN",
            64,
        )

        self.gate_hidden = env_int(
            "TGCA_V3_GATE_HIDDEN",
            32,
        )

        self.cap_min = env_float(
            "TGCA_V3_CAP_MIN",
            0.004,
        )

        self.cap_max = env_float(
            "TGCA_V3_CAP_MAX",
            0.060,
        )

        if not (
            0.0
            <= self.cap_min
            < self.cap_max
        ):
            raise ValueError({
                "cap_min": self.cap_min,
                "cap_max": self.cap_max,
            })

        self.max_signed_gate = env_float(
            "TGCA_V3_MAX_SIGNED_GATE",
            1.0,
        )

        self.max_global_scale = env_float(
            "TGCA_V3_MAX_GLOBAL_SCALE",
            1.0,
        )

        initial_scale = env_float(
            "TGCA_V3_INIT_SCALE",
            0.20,
        )

        if not (
            0.0
            < initial_scale
            < self.max_global_scale
        ):
            raise ValueError({
                "initial_scale": initial_scale,
                "max_global_scale":
                    self.max_global_scale,
            })

        ratio = (
            initial_scale
            / self.max_global_scale
        )

        self.scale_logit = nn.Parameter(
            torch.tensor(
                math.log(
                    ratio / (1.0 - ratio)
                ),
                dtype=torch.float32,
            )
        )

        # -------------------------------------------------
        # Tail–Energy expert
        # -------------------------------------------------
        self.tail_conv = nn.Sequential(
            nn.Conv1d(
                self.channels,
                self.channels,
                kernel_size=7,
                padding=3,
                groups=self.channels,
            ),
            nn.GELU(),
            nn.Conv1d(
                self.channels,
                self.channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                self.channels,
                self.channels,
                kernel_size=5,
                padding=2,
                groups=self.channels,
            ),
            nn.GELU(),
        )

        self.tail_decoder = nn.Linear(
            self.patch_tokens,
            self.pred_len,
        )

        # -------------------------------------------------
        # Phase–Horizon expert
        #
        # real + imaginary + log magnitude.
        # -------------------------------------------------
        self.phase_decoder = nn.Sequential(
            nn.Linear(
                self.freq_bins * 3,
                self.hidden,
            ),
            nn.LayerNorm(
                self.hidden,
            ),
            nn.GELU(),
            nn.Linear(
                self.hidden,
                self.pred_len,
            ),
        )

        # Risk feature dimensions:
        #
        # 0 channel kurtosis
        # 1 increment kurtosis
        # 2 increment RMS
        # 3 absolute energy drift
        # 4 phase drift
        # 5 spectral entropy
        # 6 absolute base slope
        # 7 base increment RMS
        self.risk_feature_count = 8

        self.risk_normalizer = nn.LayerNorm(
            self.risk_feature_count,
        )

        # Tail vs Phase expert routing.
        self.mechanism_gate = nn.Sequential(
            nn.Linear(
                self.risk_feature_count,
                self.gate_hidden,
            ),
            nn.GELU(),
            nn.Linear(
                self.gate_hidden,
                2,
            ),
        )

        # Signed direction by channel × horizon block.
        self.direction_head = nn.Sequential(
            nn.Linear(
                self.risk_feature_count,
                self.gate_hidden,
            ),
            nn.GELU(),
            nn.Linear(
                self.gate_hidden,
                self.direction_blocks,
            ),
        )

        # Risk-dependent cap by channel × horizon block.
        self.risk_head = nn.Sequential(
            nn.Linear(
                self.risk_feature_count,
                self.gate_hidden,
            ),
            nn.GELU(),
            nn.Linear(
                self.gate_hidden,
                self.direction_blocks,
            ),
        )

        # Channel factor in [0.5, 1.5], initialized at 1.
        self.channel_logit = nn.Parameter(
            torch.zeros(
                self.channels,
                dtype=torch.float32,
            )
        )

        self._print_count = {
            "train": 0,
            "eval": 0,
        }

        self._last_diagnostics: dict[
            str,
            Any,
        ] = {}

        self._initialize_safe_residual()

    def _initialize_safe_residual(self) -> None:
        """
        Experts retain ordinary initialization.

        Only the direction output is initialized to zero,
        making the complete model exactly equal to the
        canonical base before training.
        """
        direction_output = (
            self.direction_head[-1]
        )

        nn.init.zeros_(
            direction_output.weight
        )

        nn.init.zeros_(
            direction_output.bias
        )

        # Start risk activation conservatively:
        # sigmoid(-1.5) ~= 0.182.
        risk_output = self.risk_head[-1]

        nn.init.zeros_(
            risk_output.weight
        )

        nn.init.constant_(
            risk_output.bias,
            -1.5,
        )

    @staticmethod
    def _standardize(
        value: torch.Tensor,
    ):
        mean = value.mean(
            dim=1,
            keepdim=True,
        )

        std = value.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-4)

        normalized = (
            value - mean
        ) / std

        return normalized, mean, std

    def _risk_features(
        self,
        normalized: torch.Tensor,
        base_prediction: torch.Tensor,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
    ):
        # [B, L, C] -> [B, C, L]
        channel_first = normalized.transpose(
            1,
            2,
        )

        difference = (
            channel_first[:, :, 1:]
            - channel_first[:, :, :-1]
        )

        difference_mean = difference.mean(
            dim=-1,
            keepdim=True,
        )

        difference_centered = (
            difference
            - difference_mean
        )

        difference_variance = (
            difference_centered.pow(2)
            .mean(dim=-1)
            .clamp_min(1e-6)
        )

        difference_m4 = (
            difference_centered.pow(4)
            .mean(dim=-1)
        )

        increment_kurtosis = (
            difference_m4
            / difference_variance.pow(2)
        )

        increment_rms = torch.sqrt(
            difference.pow(2).mean(
                dim=-1
            )
            + 1e-6
        )

        # normalized input has approximately zero mean
        # and unit variance per channel.
        channel_kurtosis = (
            channel_first.pow(4)
            .mean(dim=-1)
        )

        quarter = max(
            1,
            self.seq_len // 4,
        )

        early_energy = (
            channel_first[
                :, :, :quarter
            ]
            .pow(2)
            .mean(dim=-1)
        )

        late_energy = (
            channel_first[
                :, :, -quarter:
            ]
            .pow(2)
            .mean(dim=-1)
        )

        absolute_energy_drift = (
            torch.log(
                (
                    late_energy + 1e-6
                )
                / (
                    early_energy + 1e-6
                )
            )
            .abs()
        )

        # ---------------------------------------------
        # Frequency statistics and phase drift.
        # ---------------------------------------------
        spectrum = torch.fft.rfft(
            channel_first.float(),
            dim=-1,
        )

        power = spectrum.abs().pow(2)

        if power.shape[-1] > 1:
            power_no_dc = power[
                :, :, 1:
            ]

            dominant_index = (
                power_no_dc.argmax(
                    dim=-1
                )
                + 1
            )
        else:
            dominant_index = torch.zeros(
                channel_first.shape[:2],
                device=channel_first.device,
                dtype=torch.long,
            )

        probability = power / (
            power.sum(
                dim=-1,
                keepdim=True,
            )
            + 1e-8
        )

        spectral_entropy = -(
            probability
            * torch.log(
                probability + 1e-8
            )
        ).sum(dim=-1)

        spectral_entropy = (
            spectral_entropy
            / max(
                math.log(
                    max(
                        power.shape[-1],
                        2,
                    )
                ),
                1e-6,
            )
        )

        half = self.seq_len // 2

        early_half = channel_first[
            :, :, :half
        ]

        late_half = channel_first[
            :, :, -half:
        ]

        early_spectrum = torch.fft.rfft(
            early_half.float(),
            dim=-1,
        )

        late_spectrum = torch.fft.rfft(
            late_half.float(),
            dim=-1,
        )

        half_index = torch.round(
            dominant_index.float()
            * float(half)
            / float(self.seq_len)
        ).long()

        half_index = half_index.clamp(
            min=1,
            max=max(
                early_spectrum.shape[-1]
                - 1,
                1,
            ),
        )

        gather_index = half_index.unsqueeze(
            -1
        )

        early_component = torch.gather(
            early_spectrum,
            dim=-1,
            index=gather_index,
        ).squeeze(-1)

        late_component = torch.gather(
            late_spectrum,
            dim=-1,
            index=gather_index,
        ).squeeze(-1)

        phase_delta = (
            torch.angle(late_component)
            - torch.angle(early_component)
        )

        phase_drift = torch.atan2(
            torch.sin(phase_delta),
            torch.cos(phase_delta),
        ).abs()

        # ---------------------------------------------
        # Base forecast state.
        # ---------------------------------------------
        base_normalized = (
            base_prediction
            - input_mean
        ) / input_std

        base_slope = (
            base_normalized[:, -1, :]
            - base_normalized[:, 0, :]
        ).abs() / math.sqrt(
            max(self.pred_len, 1)
        )

        base_difference = (
            base_normalized[:, 1:, :]
            - base_normalized[:, :-1, :]
        )

        base_increment_rms = torch.sqrt(
            base_difference.pow(2)
            .mean(dim=1)
            + 1e-6
        )

        risk_features = torch.stack(
            [
                channel_kurtosis,
                increment_kurtosis,
                increment_rms,
                absolute_energy_drift,
                phase_drift,
                spectral_entropy,
                base_slope,
                base_increment_rms,
            ],
            dim=-1,
        )

        risk_features = torch.nan_to_num(
            risk_features,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )

        normalized_features = (
            self.risk_normalizer(
                risk_features
            )
        )

        return (
            normalized_features,
            spectrum,
            difference,
        )

    def _tail_energy_expert(
        self,
        difference: torch.Tensor,
    ) -> torch.Tensor:
        # difference: [B, C, L-1]
        difference = F.pad(
            difference,
            (1, 0),
            mode="replicate",
        )

        encoded = self.tail_conv(
            difference
        )

        pooled = F.adaptive_avg_pool1d(
            encoded,
            self.patch_tokens,
        )

        output = self.tail_decoder(
            pooled
        )

        return output.transpose(
            1,
            2,
        )

    def _phase_horizon_expert(
        self,
        spectrum: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        spectrum = spectrum[
            :, :, :self.freq_bins
        ]

        features = torch.cat(
            [
                spectrum.real,
                spectrum.imag,
                torch.log1p(
                    spectrum.abs()
                ),
            ],
            dim=-1,
        ).to(output_dtype)

        output = self.phase_decoder(
            features
        )

        return output.transpose(
            1,
            2,
        )

    def _blockwise_cap(
        self,
        raw_residual: torch.Tensor,
        cap_blocks: torch.Tensor,
    ):
        """
        raw_residual:
            [B, H, C]

        cap_blocks:
            [B, C, K]
        """
        batch = raw_residual.shape[0]

        channel_first = raw_residual.transpose(
            1,
            2,
        ).contiguous()

        blocked = channel_first.view(
            batch,
            self.channels,
            self.direction_blocks,
            self.block_size,
        )

        block_rms = torch.sqrt(
            blocked.pow(2).mean(
                dim=-1
            )
            + 1e-12
        )

        scale = (
            cap_blocks
            / block_rms
        ).clamp(
            max=1.0,
        )

        blocked = (
            blocked
            * scale.unsqueeze(-1)
        )

        residual = blocked.reshape(
            batch,
            self.channels,
            self.pred_len,
        ).transpose(
            1,
            2,
        )

        saturation_rate = (
            scale < 0.999
        ).float().mean()

        return (
            residual,
            block_rms,
            scale,
            saturation_rate,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        base_prediction: torch.Tensor,
    ) -> torch.Tensor:
        normalized, mean, std = (
            self._standardize(
                x_enc
            )
        )

        (
            risk_features,
            spectrum,
            difference,
        ) = self._risk_features(
            normalized,
            base_prediction,
            mean,
            std,
        )

        tail_output = (
            self._tail_energy_expert(
                difference
            )
        )

        phase_output = (
            self._phase_horizon_expert(
                spectrum,
                normalized.dtype,
            )
        )

        mechanism_weights = F.softmax(
            self.mechanism_gate(
                risk_features
            ),
            dim=-1,
        )

        # ===== TGCA_V3_COUNTERFACTUAL_INFERENCE_V1 =====
        #
        # Inference-only expert override. The trained
        # checkpoint remains unchanged.
        counterfactual_expert_mode = (
            os.environ.get(
                "TGCA_V3_INFER_EXPERT_MODE",
                "learned",
            )
            .strip()
            .lower()
        )

        valid_counterfactual_modes = {
            "learned",
            "balanced",
            "phase",
            "tail",
        }

        if (
            counterfactual_expert_mode
            not in valid_counterfactual_modes
        ):
            raise ValueError(
                "TGCA_V3_INFER_EXPERT_MODE must be one of "
                f"{sorted(valid_counterfactual_modes)}, got "
                f"{counterfactual_expert_mode!r}"
            )

        if not self.training:
            if counterfactual_expert_mode == "balanced":
                mechanism_weights = torch.full_like(
                    mechanism_weights,
                    0.5,
                )

            elif counterfactual_expert_mode == "phase":
                tail_weight = torch.zeros_like(
                    mechanism_weights[..., 0]
                )

                phase_weight = torch.ones_like(
                    mechanism_weights[..., 1]
                )

                mechanism_weights = torch.stack(
                    [
                        tail_weight,
                        phase_weight,
                    ],
                    dim=-1,
                )

            elif counterfactual_expert_mode == "tail":
                tail_weight = torch.ones_like(
                    mechanism_weights[..., 0]
                )

                phase_weight = torch.zeros_like(
                    mechanism_weights[..., 1]
                )

                mechanism_weights = torch.stack(
                    [
                        tail_weight,
                        phase_weight,
                    ],
                    dim=-1,
                )

        expert_outputs = torch.stack(
            [
                tail_output,
                phase_output,
            ],
            dim=-1,
        )

        mixed_basis = (
            expert_outputs
            * mechanism_weights.unsqueeze(1)
        ).sum(dim=-1)

        mixed_basis = torch.tanh(
            mixed_basis
        )

        direction_blocks = (
            self.max_signed_gate
            * torch.tanh(
                self.direction_head(
                    risk_features
                )
            )
        )

        direction_curve = F.interpolate(
            direction_blocks,
            size=self.pred_len,
            mode="linear",
            align_corners=False,
        ).transpose(
            1,
            2,
        )

        risk_blocks = torch.sigmoid(
            self.risk_head(
                risk_features
            )
        )

        channel_scale = (
            0.5
            + torch.sigmoid(
                self.channel_logit
            )
        )

        cap_blocks = (
            self.cap_min
            + (
                self.cap_max
                - self.cap_min
            )
            * risk_blocks
            * channel_scale[
                None, :, None
            ]
        ).clamp(
            min=self.cap_min,
            max=self.cap_max,
        )

        counterfactual_fixed_cap_text = (
            os.environ.get(
                "TGCA_V3_INFER_FIXED_CAP",
                "",
            ).strip()
        )

        counterfactual_fixed_cap = None

        if (
            not self.training
            and counterfactual_fixed_cap_text
        ):
            counterfactual_fixed_cap = float(
                counterfactual_fixed_cap_text
            )

            if not (
                self.cap_min
                <= counterfactual_fixed_cap
                <= self.cap_max
            ):
                raise ValueError(
                    {
                        "fixed_cap":
                            counterfactual_fixed_cap,
                        "cap_min":
                            self.cap_min,
                        "cap_max":
                            self.cap_max,
                    }
                )

            cap_blocks = torch.full_like(
                cap_blocks,
                counterfactual_fixed_cap,
            )

        if (
            not self.training
            and not getattr(
                self,
                "_counterfactual_audit_printed",
                False,
            )
        ):
            print(
                "[TGCA_V3_COUNTERFACTUAL]",
                {
                    "expert_mode":
                        counterfactual_expert_mode,
                    "fixed_cap":
                        counterfactual_fixed_cap,
                    "checkpoint_parameters_changed":
                        False,
                },
                flush=True,
            )

            self._counterfactual_audit_printed = True

        # ===== TGCA_V3_COUNTERFACTUAL_INFERENCE_V1_END =====

        global_scale = (
            self.max_global_scale
            * torch.sigmoid(
                self.scale_logit
            )
        )

        raw_residual = (
            global_scale
            * direction_curve
            * mixed_basis
            * std
        )

        (
            residual,
            raw_block_rms,
            cap_scale,
            saturation_rate,
        ) = self._blockwise_cap(
            raw_residual,
            cap_blocks,
        )

        residual_rms = torch.sqrt(
            residual.pow(2).mean()
            + 1e-12
        )

        direction_abs_mean = (
            direction_curve.abs().mean()
        )

        direction_positive_rate = (
            direction_curve > 0.0
        ).float().mean()

        risk_mean = risk_blocks.mean()

        cap_mean = cap_blocks.mean()

        mechanism_mean = (
            mechanism_weights.mean(
                dim=(0, 1)
            )
        )

        self._last_diagnostics = {
            "residual_rms":
                residual_rms.detach(),

            "direction_abs_mean":
                direction_abs_mean.detach(),

            "direction_positive_rate":
                direction_positive_rate.detach(),

            "risk_mean":
                risk_mean.detach(),

            "cap_mean":
                cap_mean.detach(),

            "cap_min_batch":
                cap_blocks.min().detach(),

            "cap_max_batch":
                cap_blocks.max().detach(),

            "saturation_rate":
                saturation_rate.detach(),

            "mechanism_mean":
                mechanism_mean.detach(),

            "channel_scale":
                channel_scale.detach(),

            "raw_block_rms_mean":
                raw_block_rms.mean().detach(),

            "cap_scale_mean":
                cap_scale.mean().detach(),
        }

        mode_name = (
            "train"
            if self.training
            else "eval"
        )

        if self._print_count[mode_name] < 5:
            print(
                "[TGCA_MECHANISM_V3_FORWARD]",
                {
                    "training":
                        bool(self.training),

                    "global_scale":
                        float(
                            global_scale.detach()
                            .cpu()
                        ),

                    "residual_rms":
                        float(
                            residual_rms.detach()
                            .cpu()
                        ),

                    "direction_abs_mean":
                        float(
                            direction_abs_mean
                            .detach().cpu()
                        ),

                    "direction_positive_rate":
                        float(
                            direction_positive_rate
                            .detach().cpu()
                        ),

                    "risk_mean":
                        float(
                            risk_mean.detach()
                            .cpu()
                        ),

                    "cap_mean":
                        float(
                            cap_mean.detach()
                            .cpu()
                        ),

                    "cap_min_batch":
                        float(
                            cap_blocks.min()
                            .detach().cpu()
                        ),

                    "cap_max_batch":
                        float(
                            cap_blocks.max()
                            .detach().cpu()
                        ),

                    "saturation_rate":
                        float(
                            saturation_rate
                            .detach().cpu()
                        ),

                    "mechanism_mean":
                        mechanism_mean.detach()
                        .cpu()
                        .tolist(),

                    "channel_scale":
                        channel_scale.detach()
                        .cpu()
                        .tolist(),
                },
                flush=True,
            )

            self._print_count[
                mode_name
            ] += 1

        return (
            base_prediction + residual
        )

    def diagnostics(self):
        return self._last_diagnostics


class Model(nn.Module):
    """
    Frozen canonical TGCA plus mechanism-driven correction
    field using a longer lookback.
    """

    def __init__(self, configs) -> None:
        super().__init__()

        self.base_seq_len = env_int(
            "TGCA_V3_BASE_SEQ_LEN",
            96,
        )

        self.freeze_base = env_bool(
            "TGCA_V3_FREEZE_BASE",
            True,
        )

        if int(configs.seq_len) < self.base_seq_len:
            raise ValueError({
                "seq_len":
                    int(configs.seq_len),
                "base_seq_len":
                    self.base_seq_len,
            })

        base_configs = copy.copy(
            configs
        )

        base_configs.seq_len = (
            self.base_seq_len
        )

        self.base = TGCA_Forecaster.Model(
            base_configs
        )

        self.adapter = (
            RiskConditionalMechanismAdapter(
                configs
            )
        )

        if self.freeze_base:
            self.base.requires_grad_(
                False
            )
            self.base.eval()

        total_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        print(
            "[TGCA_MECHANISM_V3_INIT]",
            {
                "base_seq_len":
                    self.base_seq_len,

                "adapter_seq_len":
                    int(configs.seq_len),

                "pred_len":
                    int(configs.pred_len),

                "freeze_base":
                    self.freeze_base,

                "cap_min":
                    self.adapter.cap_min,

                "cap_max":
                    self.adapter.cap_max,

                "direction_blocks":
                    self.adapter.direction_blocks,

                "total_parameters":
                    total_parameters,

                "trainable_parameters":
                    trainable_parameters,
            },
            flush=True,
        )

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)

        if self.freeze_base:
            self.base.eval()

        return self

    @staticmethod
    def _prediction_from_output(output):
        if isinstance(
            output,
            (tuple, list),
        ):
            return output[0]

        return output

    @staticmethod
    def _replace_prediction(
        output,
        prediction,
    ):
        if isinstance(output, tuple):
            return (
                prediction,
                *output[1:],
            )

        if isinstance(output, list):
            return [
                prediction,
                *output[1:],
            ]

        return prediction

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
    ):
        base_x = x_enc[
            :, -self.base_seq_len:, :
        ]

        if x_mark_enc is None:
            base_x_mark = None
        else:
            base_x_mark = x_mark_enc[
                :, -self.base_seq_len:, :
            ]

        context = (
            torch.no_grad()
            if self.freeze_base
            else nullcontext()
        )

        with context:
            base_output = self.base(
                base_x,
                base_x_mark,
                x_dec,
                x_mark_dec,
                mask,
            )

        base_prediction = (
            self._prediction_from_output(
                base_output
            )
        )

        prediction = self.adapter(
            x_enc,
            base_prediction,
        )

        return self._replace_prediction(
            base_output,
            prediction,
        )

    def auxiliary_loss(self):
        return (
            self.adapter.scale_logit
            * 0.0
        )

    def diagnostics(self):
        return self.adapter.diagnostics()
