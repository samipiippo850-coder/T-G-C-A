from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import TGCA_CPTRD


env_float = TGCA_CPTRD.env_float
env_bool = TGCA_CPTRD.env_bool


class SignedProxyDynamics(
    TGCA_CPTRD.CoupledParallelTensorDynamics
):
    """
    CPTRD-V4

    1. 使用 core.state.proxy_conv 作为动态观测；
    2. raw delta 只负责方向；
    3. signed amplitude 负责正负幅度；
    4. confidence 决定是否采用修正。
    """

    def __init__(self, configs) -> None:
        super().__init__(configs)

        self.amp_max = env_float(
            "CPTRD_V4_AMP_MAX",
            0.50,
        )

        self.amp_deadzone = env_float(
            "CPTRD_V4_AMP_DEADZONE",
            0.002,
        )

        self.direction_eps = env_float(
            "CPTRD_V4_DIRECTION_EPS",
            1e-6,
        )

        self.proxy_encoder = nn.Sequential(
            nn.Linear(
                self.c_out * 6,
                self.hidden_dim,
            ),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
        )

        common_dim = (
            self.hidden_dim * 5
            + self.latent_dim
        )

        # 每个通道均可拥有不同的正负幅度。
        self.signed_amp_head = nn.Linear(
            common_dim,
            self.c_out,
        )

        # 每个 horizon block 一个置信度。
        self.confidence_head = nn.Linear(
            common_dim,
            1,
        )

        # 初始幅度严格为 0：初始输出等于基础模型。
        nn.init.zeros_(
            self.signed_amp_head.weight
        )
        nn.init.zeros_(
            self.signed_amp_head.bias
        )

        # sigmoid(-1.386) ≈ 0.20
        nn.init.zeros_(
            self.confidence_head.weight
        )
        nn.init.constant_(
            self.confidence_head.bias,
            -1.386,
        )

        self._last_directions = []
        self._last_amp_raw = []
        self._last_amplitudes = []
        self._last_confidences = []
        self._last_ranges = []

    def _shape_features(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        sequence: [B,T,C]
        output:   [B,6C]
        """
        mean = sequence.mean(dim=1)

        std = sequence.std(
            dim=1,
            unbiased=False,
        )

        first = sequence[:, 0, :]
        last = sequence[:, -1, :]
        slope = last - first

        if sequence.shape[1] > 1:
            difference = (
                sequence[:, 1:, :]
                - sequence[:, :-1, :]
            )

            diff_energy = (
                difference
                .abs()
                .mean(dim=1)
            )
        else:
            diff_energy = torch.zeros_like(
                mean
            )

        return torch.cat(
            [
                mean,
                std,
                first,
                last,
                slope,
                diff_energy,
            ],
            dim=-1,
        )

    def _normalize_proxy(
        self,
        proxy_state: torch.Tensor,
    ) -> torch.Tensor:
        if proxy_state.ndim != 3:
            raise RuntimeError(
                "proxy_conv must be rank-3, "
                f"got {tuple(proxy_state.shape)}"
            )

        # 常见形状：[B,C,H]
        if (
            proxy_state.shape[1]
            == self.c_out
            and proxy_state.shape[2]
            == self.pred_len
        ):
            return proxy_state.transpose(1, 2)

        # 已经是 [B,H,C]
        if (
            proxy_state.shape[1]
            == self.pred_len
            and proxy_state.shape[2]
            == self.c_out
        ):
            return proxy_state

        raise RuntimeError(
            "Unexpected proxy_conv shape: "
            f"{tuple(proxy_state.shape)}, "
            f"expected [B,{self.c_out},{self.pred_len}] "
            f"or [B,{self.pred_len},{self.c_out}]"
        )

    def forward(
        self,
        base_pred: torch.Tensor,
        x_enc: torch.Tensor,
        proxy_state: torch.Tensor,
    ) -> torch.Tensor:
        proxy_state = self._normalize_proxy(
            proxy_state
        )

        batch_size, horizon, channels = (
            base_pred.shape
        )

        if horizon != self.pred_len:
            raise RuntimeError(
                f"pred_len mismatch: {horizon}"
            )

        if channels != self.c_out:
            raise RuntimeError(
                f"c_out mismatch: {channels}"
            )

        context = self._context(x_enc)

        z = torch.tanh(
            self.z_init(context)
        )

        states = tuple(
            torch.tanh(
                initializer(context)
            )
            for initializer
            in self.state_initializers
        )

        correction_blocks = []

        self._last_directions = []
        self._last_amp_raw = []
        self._last_amplitudes = []
        self._last_confidences = []
        self._last_ranges = []

        tau_values = []
        event_values = []
        amplitude_values = []
        confidence_values = []

        for block_index in range(
            self.n_blocks
        ):
            left = self.block_edges[
                block_index
            ]

            right = self.block_edges[
                block_index + 1
            ]

            base_block = base_pred[
                :,
                left:right,
                :,
            ]

            proxy_block = proxy_state[
                :,
                left:right,
                :,
            ]

            base_features = (
                self._shape_features(
                    base_block
                )
            )

            proxy_features = (
                self._shape_features(
                    proxy_block
                )
            )

            phi = torch.tanh(
                context
                + self.block_encoder(
                    base_features
                )
                + self.proxy_encoder(
                    proxy_features
                )
                + self.z_to_phi(z)
                + self.position_embedding[
                    block_index
                ].unsqueeze(0)
            )

            old_states = states

            states = tuple(
                self._update_state(
                    state_index,
                    phi,
                    old_states,
                )
                for state_index in range(4)
            )

            h_s, h_d, h_tau, h_r = states

            (
                stable,
                drift,
                tau,
                closure,
            ) = self._operators(
                h_s,
                h_d,
                h_tau,
                h_r,
            )

            common_before = torch.cat(
                [
                    phi,
                    h_s,
                    h_d,
                    h_tau,
                    h_r,
                    z,
                ],
                dim=-1,
            )

            # 仍只控制动力学中的 closure。
            event_gate = torch.sigmoid(
                self.event_gate_head(
                    common_before
                )
            ).view(
                batch_size,
                1,
                1,
            )

            z_proposed = self._dynamics_step(
                z,
                stable,
                drift,
                closure,
                event_gate,
            )

            common_update = torch.cat(
                [
                    phi,
                    h_s,
                    h_d,
                    h_tau,
                    h_r,
                    z_proposed,
                ],
                dim=-1,
            )

            update_gate = torch.sigmoid(
                self.update_gate_head(
                    common_update
                )
            )

            z = (
                (1.0 - update_gate) * z
                + update_gate * z_proposed
            )

            readout = torch.cat(
                [
                    phi,
                    h_s,
                    h_d,
                    h_tau,
                    h_r,
                    z,
                ],
                dim=-1,
            )

            raw_direction = (
                self.output_head(readout)
                .reshape(
                    batch_size,
                    self.block_size,
                    self.c_out,
                )
            )

            actual_length = right - left

            raw_direction = raw_direction[
                :,
                :actual_length,
                :,
            ]

            # 每个样本、每个通道在时间维归一化。
            direction_rms = (
                raw_direction
                .pow(2)
                .mean(
                    dim=1,
                    keepdim=True,
                )
                .add(
                    self.direction_eps
                )
                .sqrt()
            )

            direction = (
                raw_direction
                / direction_rms
            )

            amp_raw = self.signed_amp_head(
                readout
            )

            signed_amplitude = (
                self.amp_max
                * torch.tanh(
                    amp_raw
                )
            )

            confidence = torch.sigmoid(
                self.confidence_head(
                    readout
                )
            )

            # 不再乘 residual_scale、output_gate、
            # event_gate。方向与幅度彻底分离。
            block_correction = (
                direction
                * signed_amplitude.unsqueeze(1)
                * confidence.unsqueeze(1)
            )

            correction_blocks.append(
                block_correction
            )

            self._last_directions.append(
                direction
            )
            self._last_amp_raw.append(
                amp_raw
            )
            self._last_amplitudes.append(
                signed_amplitude
            )
            self._last_confidences.append(
                confidence
            )
            self._last_ranges.append(
                (left, right)
            )

            tau_values.append(
                tau.mean()
            )
            event_values.append(
                event_gate.mean()
            )
            amplitude_values.append(
                signed_amplitude
                .abs()
                .mean()
            )
            confidence_values.append(
                confidence.mean()
            )

        correction = torch.cat(
            correction_blocks,
            dim=1,
        )

        self._last_base_pred = base_pred
        self._last_correction = correction

        self._last_auxiliary_loss = (
            1e-5
            * correction.pow(2).mean()
        )

        self._last_diagnostics = {
            "tau_mean": (
                torch.stack(
                    tau_values
                ).mean().detach()
            ),
            "event_rate": (
                torch.stack(
                    event_values
                ).mean().detach()
            ),
            "signed_amp_abs": (
                torch.stack(
                    amplitude_values
                ).mean().detach()
            ),
            "confidence": (
                torch.stack(
                    confidence_values
                ).mean().detach()
            ),
            "correction_rms": (
                correction
                .pow(2)
                .mean()
                .sqrt()
                .detach()
            ),
        }

        return base_pred + correction

    def supervision_loss(
        self,
        target: torch.Tensor,
    ):
        if (
            self._last_base_pred is None
            or not self._last_directions
        ):
            zero = (
                self.signed_amp_head.weight
                .sum()
                * 0.0
            )
            return zero, {}

        amp_losses = []
        sign_losses = []
        confidence_losses = []
        direction_losses = []

        oracle_safe_rates = []
        sign_accuracies = []
        confidence_accuracies = []
        oracle_amp_values = []

        for (
            direction,
            amp_raw,
            amplitude,
            confidence,
            block_range,
        ) in zip(
            self._last_directions,
            self._last_amp_raw,
            self._last_amplitudes,
            self._last_confidences,
            self._last_ranges,
        ):
            left, right = block_range

            base_block = (
                self._last_base_pred[
                    :,
                    left:right,
                    :,
                ]
            )

            target_block = target[
                :,
                left:right,
                :,
            ]

            desired_correction = (
                target_block
                - base_block
            ).detach()

            denominator = (
                direction
                .pow(2)
                .mean(dim=1)
                .clamp_min(1e-6)
            )

            # 给定方向时，MSE 下最优带符号幅度。
            oracle_amplitude = (
                (
                    desired_correction
                    * direction
                )
                .mean(dim=1)
                / denominator
            )

            oracle_amplitude = torch.clamp(
                oracle_amplitude,
                min=-self.amp_max,
                max=self.amp_max,
            )

            amp_loss = F.smooth_l1_loss(
                amplitude,
                oracle_amplitude,
            )

            active_sign = (
                oracle_amplitude.abs()
                > self.amp_deadzone
            )

            if active_sign.any():
                sign_target = (
                    oracle_amplitude > 0
                ).float()

                sign_loss = (
                    F.binary_cross_entropy_with_logits(
                        amp_raw[
                            active_sign
                        ],
                        sign_target[
                            active_sign
                        ],
                    )
                )

                sign_accuracy = (
                    (
                        (amp_raw > 0)
                        == (
                            oracle_amplitude > 0
                        )
                    )[
                        active_sign
                    ]
                    .float()
                    .mean()
                )
            else:
                sign_loss = (
                    amp_raw.sum() * 0.0
                )
                sign_accuracy = (
                    amp_raw.new_tensor(1.0)
                )

            oracle_correction = (
                direction
                * oracle_amplitude.unsqueeze(1)
            )

            base_error = (
                base_block
                - target_block
            )

            oracle_error = (
                base_block
                + oracle_correction
                - target_block
            )

            base_mse = (
                base_error
                .pow(2)
                .mean(dim=(1, 2))
            )

            oracle_mse = (
                oracle_error
                .pow(2)
                .mean(dim=(1, 2))
            )

            base_mae = (
                base_error
                .abs()
                .mean(dim=(1, 2))
            )

            oracle_mae = (
                oracle_error
                .abs()
                .mean(dim=(1, 2))
            )

            # 只有 MSE 与 MAE 同时改善才值得启用。
            safe_target = (
                (oracle_mse < base_mse)
                & (oracle_mae <= base_mae)
            ).float()

            confidence_loss = (
                F.binary_cross_entropy(
                    confidence.squeeze(-1),
                    safe_target,
                )
            )

            confidence_prediction = (
                confidence.squeeze(-1)
                >= 0.5
            )

            confidence_accuracy = (
                (
                    confidence_prediction
                    == safe_target.bool()
                )
                .float()
                .mean()
            )

            flat_direction = direction.reshape(
                direction.shape[0],
                -1,
            )

            flat_target = (
                desired_correction.reshape(
                    desired_correction.shape[0],
                    -1,
                )
            )

            direction_cosine = (
                F.cosine_similarity(
                    flat_direction,
                    flat_target,
                    dim=1,
                    eps=1e-8,
                )
            )

            direction_loss = (
                1.0
                - direction_cosine.mean()
            )

            amp_losses.append(amp_loss)
            sign_losses.append(sign_loss)
            confidence_losses.append(
                confidence_loss
            )
            direction_losses.append(
                direction_loss
            )

            oracle_safe_rates.append(
                safe_target.mean()
            )
            sign_accuracies.append(
                sign_accuracy
            )
            confidence_accuracies.append(
                confidence_accuracy
            )
            oracle_amp_values.append(
                oracle_amplitude
                .abs()
                .mean()
            )

        amp_loss = torch.stack(
            amp_losses
        ).mean()

        sign_loss = torch.stack(
            sign_losses
        ).mean()

        confidence_loss = torch.stack(
            confidence_losses
        ).mean()

        direction_loss = torch.stack(
            direction_losses
        ).mean()

        lambda_amp = env_float(
            "CPTRD_V4_LAMBDA_AMP",
            0.50,
        )

        lambda_sign = env_float(
            "CPTRD_V4_LAMBDA_SIGN",
            0.10,
        )

        lambda_confidence = env_float(
            "CPTRD_V4_LAMBDA_CONFIDENCE",
            0.10,
        )

        lambda_direction = env_float(
            "CPTRD_V4_LAMBDA_DIRECTION",
            0.02,
        )

        total = (
            lambda_amp * amp_loss
            + lambda_sign * sign_loss
            + lambda_confidence
            * confidence_loss
            + lambda_direction
            * direction_loss
        )

        statistics = {
            "amp_loss": amp_loss.detach(),
            "sign_loss": sign_loss.detach(),
            "confidence_loss": (
                confidence_loss.detach()
            ),
            "direction_loss": (
                direction_loss.detach()
            ),
            "oracle_amp_abs": (
                torch.stack(
                    oracle_amp_values
                ).mean().detach()
            ),
            "oracle_safe_rate": (
                torch.stack(
                    oracle_safe_rates
                ).mean().detach()
            ),
            "sign_accuracy": (
                torch.stack(
                    sign_accuracies
                ).mean().detach()
            ),
            "confidence_accuracy": (
                torch.stack(
                    confidence_accuracies
                ).mean().detach()
            ),
        }

        return total, statistics


class Model(TGCA_CPTRD.Model):
    def __init__(self, configs) -> None:
        super().__init__(configs)

        self.cptrd = SignedProxyDynamics(
            configs
        )

        self._proxy_cache = None
        self._proxy_hook_handle = None

        named_modules = dict(
            self.base.named_modules()
        )

        proxy_name = None

        if "core.state.proxy_conv" in named_modules:
            proxy_name = (
                "core.state.proxy_conv"
            )
        else:
            candidates = [
                name
                for name in named_modules
                if name.endswith(
                    "proxy_conv"
                )
            ]

            if candidates:
                proxy_name = candidates[0]

        if proxy_name is None:
            raise RuntimeError(
                "Could not find proxy_conv "
                "inside TGCA_Forecaster."
            )

        proxy_module = named_modules[
            proxy_name
        ]

        self._proxy_hook_handle = (
            proxy_module.register_forward_hook(
                self._capture_proxy
            )
        )

        print(
            "[CPTRD_V4] "
            f"proxy_module={proxy_name} "
            f"amp_max={self.cptrd.amp_max}"
        )

    def _capture_proxy(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]

        if not torch.is_tensor(output):
            raise RuntimeError(
                "proxy_conv output is not Tensor"
            )

        self._proxy_cache = output

    def forward(
        self,
        x_enc: torch.Tensor,
        *args,
        **kwargs,
    ):
        self._proxy_cache = None

        if self.freeze_base:
            with torch.no_grad():
                base_output = self.base(
                    x_enc,
                    *args,
                    **kwargs,
                )
        else:
            base_output = self.base(
                x_enc,
                *args,
                **kwargs,
            )

        if self._proxy_cache is None:
            raise RuntimeError(
                "proxy_conv hook produced no output"
            )

        if isinstance(base_output, tuple):
            corrected = self.cptrd(
                base_output[0],
                x_enc,
                self._proxy_cache,
            )
            return (
                corrected,
                *base_output[1:],
            )

        if isinstance(base_output, list):
            corrected = self.cptrd(
                base_output[0],
                x_enc,
                self._proxy_cache,
            )
            return [
                corrected,
                *base_output[1:],
            ]

        return self.cptrd(
            base_output,
            x_enc,
            self._proxy_cache,
        )
