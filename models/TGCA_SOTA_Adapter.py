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


class HeterogeneousResidualAdapter(nn.Module):
    """
    Frozen-TGCA residual adapter.

    Experts:
      1. shared trend projection
      2. depthwise temporal convolution
      3. Fourier feature decoder

    Routing features are calculated only from the available
    lookback and base prediction. No target information enters
    the router.
    """

    def __init__(self, configs) -> None:
        super().__init__()

        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.c_out)

        self.patch_tokens = min(
            env_int(
                "TGCA_ADAPTER_PATCH_TOKENS",
                24,
            ),
            self.seq_len,
        )

        self.freq_bins = min(
            env_int(
                "TGCA_ADAPTER_FREQ_BINS",
                49,
            ),
            self.seq_len // 2 + 1,
        )

        self.spectral_hidden = env_int(
            "TGCA_ADAPTER_SPECTRAL_HIDDEN",
            64,
        )

        self.gate_hidden = env_int(
            "TGCA_ADAPTER_GATE_HIDDEN",
            32,
        )

        self.max_rms = env_float(
            "TGCA_ADAPTER_MAX_RMS",
            0.20,
        )

        self.max_scale = env_float(
            "TGCA_ADAPTER_MAX_SCALE",
            1.0,
        )

        initial_scale = env_float(
            "TGCA_ADAPTER_INIT_SCALE",
            0.05,
        )

        if not (
            0.0 < initial_scale < self.max_scale
        ):
            raise ValueError(
                {
                    "initial_scale": initial_scale,
                    "max_scale": self.max_scale,
                }
            )

        scale_ratio = (
            initial_scale / self.max_scale
        )

        initial_logit = math.log(
            scale_ratio / (1.0 - scale_ratio)
        )

        self.scale_logit = nn.Parameter(
            torch.tensor(
                initial_logit,
                dtype=torch.float32,
            )
        )

        # ----------------------------------------------------
        # Expert 1: shared channel-independent trend mapping.
        # ----------------------------------------------------
        self.trend_decoder = nn.Linear(
            self.seq_len,
            self.pred_len,
        )

        # ----------------------------------------------------
        # Expert 2: local temporal shape expert.
        # ----------------------------------------------------
        self.local_conv = nn.Sequential(
            nn.Conv1d(
                self.channels,
                self.channels,
                kernel_size=5,
                padding=2,
                groups=self.channels,
            ),
            nn.GELU(),
            nn.Conv1d(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                groups=self.channels,
            ),
            nn.GELU(),
        )

        self.local_decoder = nn.Linear(
            self.patch_tokens,
            self.pred_len,
        )

        # ----------------------------------------------------
        # Expert 3: Fourier real/imaginary feature decoder.
        # ----------------------------------------------------
        self.spectral_decoder = nn.Sequential(
            nn.Linear(
                self.freq_bins * 2,
                self.spectral_hidden,
            ),
            nn.LayerNorm(
                self.spectral_hidden,
            ),
            nn.GELU(),
            nn.Linear(
                self.spectral_hidden,
                self.pred_len,
            ),
        )

        # Per-channel moment/regime gate.
        #
        # Features:
        # last z-score
        # normalized slope
        # first-difference RMS
        # skewness
        # kurtosis
        # late/early energy log-ratio
        # base forecast slope
        # base forecast difference RMS
        self.gate = nn.Sequential(
            nn.Linear(
                8,
                self.gate_hidden,
            ),
            nn.LayerNorm(
                self.gate_hidden,
            ),
            nn.GELU(),
            nn.Linear(
                self.gate_hidden,
                3,
            ),
        )

        self._print_count = 0
        self._last_diagnostics: dict[str, Any] = {}

        self._initialize_residual_to_zero()

    def _initialize_residual_to_zero(self) -> None:
        """
        The initial adapter output is exactly zero, so loading
        the canonical checkpoint reproduces the canonical result.
        """
        nn.init.zeros_(
            self.trend_decoder.weight
        )
        nn.init.zeros_(
            self.trend_decoder.bias
        )

        nn.init.zeros_(
            self.local_decoder.weight
        )
        nn.init.zeros_(
            self.local_decoder.bias
        )

        final_spectral = self.spectral_decoder[-1]

        nn.init.zeros_(
            final_spectral.weight
        )
        nn.init.zeros_(
            final_spectral.bias
        )

        final_gate = self.gate[-1]

        nn.init.zeros_(
            final_gate.weight
        )
        nn.init.zeros_(
            final_gate.bias
        )

    @staticmethod
    def _safe_standardize(
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

    def _moment_gate(
        self,
        normalized_input: torch.Tensor,
        base_prediction: torch.Tensor,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
    ) -> torch.Tensor:
        # [B, L, C] -> [B, C, L]
        channel_first = normalized_input.transpose(
            1,
            2,
        )

        difference = (
            channel_first[:, :, 1:]
            - channel_first[:, :, :-1]
        )

        last_value = channel_first[:, :, -1]

        slope = (
            channel_first[:, :, -1]
            - channel_first[:, :, 0]
        ) / math.sqrt(
            max(self.seq_len, 1)
        )

        difference_rms = torch.sqrt(
            difference.pow(2).mean(
                dim=-1
            ) + 1e-6
        )

        skewness = channel_first.pow(3).mean(
            dim=-1
        )

        kurtosis = channel_first.pow(4).mean(
            dim=-1
        )

        quarter = max(
            1,
            self.seq_len // 4,
        )

        early_energy = channel_first[
            :, :, :quarter
        ].pow(2).mean(
            dim=-1
        )

        late_energy = channel_first[
            :, :, -quarter:
        ].pow(2).mean(
            dim=-1
        )

        late_early_log_energy = torch.log(
            (
                late_energy + 1e-6
            )
            / (
                early_energy + 1e-6
            )
        )

        base_normalized = (
            base_prediction - input_mean
        ) / input_std

        base_slope = (
            base_normalized[:, -1, :]
            - base_normalized[:, 0, :]
        ) / math.sqrt(
            max(self.pred_len, 1)
        )

        base_difference = (
            base_normalized[:, 1:, :]
            - base_normalized[:, :-1, :]
        )

        base_difference_rms = torch.sqrt(
            base_difference.pow(2).mean(
                dim=1
            ) + 1e-6
        )

        gate_features = torch.stack(
            [
                last_value,
                slope,
                difference_rms,
                skewness,
                kurtosis,
                late_early_log_energy,
                base_slope,
                base_difference_rms,
            ],
            dim=-1,
        )

        gate_logits = self.gate(
            gate_features
        )

        return F.softmax(
            gate_logits,
            dim=-1,
        )

    def _trend_expert(
        self,
        normalized_input: torch.Tensor,
    ) -> torch.Tensor:
        channel_first = normalized_input.transpose(
            1,
            2,
        )

        output = self.trend_decoder(
            channel_first
        )

        return output.transpose(
            1,
            2,
        )

    def _local_expert(
        self,
        normalized_input: torch.Tensor,
    ) -> torch.Tensor:
        channel_first = normalized_input.transpose(
            1,
            2,
        )

        encoded = self.local_conv(
            channel_first
        )

        pooled = F.adaptive_avg_pool1d(
            encoded,
            self.patch_tokens,
        )

        output = self.local_decoder(
            pooled
        )

        return output.transpose(
            1,
            2,
        )

    def _spectral_expert(
        self,
        normalized_input: torch.Tensor,
    ) -> torch.Tensor:
        channel_first = normalized_input.transpose(
            1,
            2,
        )

        spectrum = torch.fft.rfft(
            channel_first.float(),
            dim=-1,
        )

        spectrum = spectrum[
            :, :, :self.freq_bins
        ]

        spectral_features = torch.cat(
            [
                spectrum.real,
                spectrum.imag,
            ],
            dim=-1,
        ).to(
            normalized_input.dtype
        )

        output = self.spectral_decoder(
            spectral_features
        )

        return output.transpose(
            1,
            2,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        base_prediction: torch.Tensor,
    ) -> torch.Tensor:
        if x_enc.ndim != 3:
            raise ValueError(
                f"x_enc must be [B,L,C], got {x_enc.shape}"
            )

        if base_prediction.ndim != 3:
            raise ValueError(
                "base_prediction must be [B,H,C], "
                f"got {base_prediction.shape}"
            )

        if x_enc.shape[1] != self.seq_len:
            raise ValueError(
                {
                    "input_seq_len":
                        int(x_enc.shape[1]),
                    "configured_seq_len":
                        self.seq_len,
                }
            )

        normalized, mean, std = (
            self._safe_standardize(
                x_enc
            )
        )

        trend = self._trend_expert(
            normalized
        )

        local = self._local_expert(
            normalized
        )

        spectral = self._spectral_expert(
            normalized
        )

        # [B, H, C, E]
        expert_outputs = torch.stack(
            [
                trend,
                local,
                spectral,
            ],
            dim=-1,
        )

        # [B, C, E]
        gate_weights = self._moment_gate(
            normalized,
            base_prediction,
            mean,
            std,
        )

        mixed_normalized = (
            expert_outputs
            * gate_weights.unsqueeze(1)
        ).sum(
            dim=-1
        )

        adapter_scale = (
            self.max_scale
            * torch.sigmoid(
                self.scale_logit
            )
        )

        residual = (
            adapter_scale
            * mixed_normalized
            * std
        )

        residual_rms = torch.sqrt(
            residual.pow(2).mean(
                dim=(1, 2),
                keepdim=True,
            ) + 1e-12
        )

        cap_factor = (
            self.max_rms
            / residual_rms
        ).clamp(
            max=1.0,
        )

        residual = (
            residual * cap_factor
        )

        final_rms = torch.sqrt(
            residual.pow(2).mean()
            + 1e-12
        )

        gate_mean = gate_weights.mean(
            dim=(0, 1)
        )

        self._last_diagnostics = {
            "adapter_scale":
                adapter_scale.detach(),
            "residual_rms":
                final_rms.detach(),
            "gate_mean":
                gate_mean.detach(),
        }

        if self._print_count < 5:
            print(
                "[TGCA_SOTA_ADAPTER_FORWARD]",
                {
                    "training":
                        bool(self.training),
                    "adapter_scale":
                        float(
                            adapter_scale.detach().cpu()
                        ),
                    "residual_rms":
                        float(
                            final_rms.detach().cpu()
                        ),
                    "gate_mean":
                        gate_mean.detach()
                        .cpu()
                        .tolist(),
                    "max_rms":
                        self.max_rms,
                },
                flush=True,
            )

            self._print_count += 1

        return (
            base_prediction + residual
        )

    def diagnostics(self):
        return self._last_diagnostics


class Model(nn.Module):
    """
    Canonical TGCA + independent residual experts.

    A longer adapter lookback can later be used while the
    canonical base continues to receive its original last
    96 observations.
    """

    def __init__(self, configs) -> None:
        super().__init__()

        self.base_seq_len = env_int(
            "TGCA_ADAPTER_BASE_SEQ_LEN",
            96,
        )

        self.freeze_base = env_bool(
            "TGCA_ADAPTER_FREEZE_BASE",
            True,
        )

        if int(configs.seq_len) < self.base_seq_len:
            raise ValueError(
                {
                    "adapter_seq_len":
                        int(configs.seq_len),
                    "base_seq_len":
                        self.base_seq_len,
                }
            )

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
            HeterogeneousResidualAdapter(
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
            "[TGCA_SOTA_ADAPTER_INIT]",
            {
                "base_seq_len":
                    self.base_seq_len,
                "adapter_seq_len":
                    int(configs.seq_len),
                "pred_len":
                    int(configs.pred_len),
                "freeze_base":
                    self.freeze_base,
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
    def _prediction_from_output(
        output,
    ):
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

        if x_mark_enc is not None:
            base_x_mark = x_mark_enc[
                :, -self.base_seq_len:, :
            ]
        else:
            base_x_mark = None

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

        corrected_prediction = self.adapter(
            x_enc,
            base_prediction,
        )

        return self._replace_prediction(
            base_output,
            corrected_prediction,
        )

    def auxiliary_loss(self):
        # Keeps compatibility with wrappers that query an
        # auxiliary model loss. The first version uses only
        # the forecasting criterion.
        return (
            self.adapter.scale_logit
            * 0.0
        )

    def diagnostics(self):
        return self.adapter.diagnostics()
