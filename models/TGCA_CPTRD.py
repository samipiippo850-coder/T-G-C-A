from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import TGCA_Forecaster


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0")
    return raw.lower() in {"1", "true", "yes", "on"}


class CoupledParallelTensorDynamics(nn.Module):
    """
    CPTRD-V1

    四个并行耦合递归状态：
        h_s   稳定状态
        h_d   漂移状态
        h_tau 弛豫状态
        h_r   闭合残差状态

    每个 horizon block 生成：
        S_k   正半定稳定张量
        D_k   反对称漂移张量
        tau_k 模态弛豫时间
        R_k   低秩闭合残差
    """

    def __init__(self, configs) -> None:
        super().__init__()

        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.c_out = int(configs.c_out)

        self.n_blocks = min(
            env_int("CPTRD_BLOCKS", 24),
            self.pred_len,
        )
        self.latent_dim = env_int("CPTRD_LATENT_DIM", 16)
        self.hidden_dim = env_int("CPTRD_HIDDEN_DIM", 48)

        self.stable_rank = env_int("CPTRD_STABLE_RANK", 4)
        self.drift_rank = env_int("CPTRD_DRIFT_RANK", 4)
        self.closure_rank = env_int("CPTRD_CLOSURE_RANK", 2)

        self.dt = env_float("CPTRD_DT", 0.5)
        self.tau_min = env_float("CPTRD_TAU_MIN", 0.5)
        self.tau_max = env_float("CPTRD_TAU_MAX", 16.0)

        self.stable_scale = env_float("CPTRD_STABLE_SCALE", 0.40)
        self.drift_scale = env_float("CPTRD_DRIFT_SCALE", 0.20)
        self.closure_scale = env_float("CPTRD_CLOSURE_SCALE", 0.05)

        self.block_size = int(
            math.ceil(self.pred_len / self.n_blocks)
        )

        edges = [
            round(i * self.pred_len / self.n_blocks)
            for i in range(self.n_blocks + 1)
        ]
        edges[0] = 0
        edges[-1] = self.pred_len
        self.block_edges = edges

        # 输入历史的当前 regime 摘要。
        self.context_encoder = nn.Sequential(
            nn.Linear(self.enc_in * 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 每个预测 block 的基础预测摘要。
        # 每个 horizon block 使用六类形状统计：
        # mean, std, first, last, slope, first-difference energy。
        self.block_encoder = nn.Sequential(
            nn.Linear(
                self.c_out * 6,
                self.hidden_dim,
            ),
            nn.LayerNorm(
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
        )

        self.position_embedding = nn.Parameter(
            torch.randn(self.n_blocks, self.hidden_dim) * 0.02
        )

        self.z_init = nn.Linear(self.hidden_dim, self.latent_dim)
        self.z_to_phi = nn.Linear(self.latent_dim, self.hidden_dim)

        # S、D、tau、R 的四个初始状态。
        self.state_initializers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(4)
        ])

        # 四条并行 GRU 状态流。
        self.cells = nn.ModuleList([
            nn.GRUCell(self.hidden_dim * 2, self.hidden_dim)
            for _ in range(4)
        ])

        # 每条状态流接收其他三条流的信息。
        self.message_projections = nn.ModuleList([
            nn.Linear(self.hidden_dim * 3, self.hidden_dim)
            for _ in range(4)
        ])

        self.message_gates = nn.ModuleList([
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
            for _ in range(4)
        ])

        self.stable_head = nn.Linear(
            self.hidden_dim,
            self.latent_dim * self.stable_rank,
        )

        self.drift_u_head = nn.Linear(
            self.hidden_dim,
            self.latent_dim * self.drift_rank,
        )
        self.drift_v_head = nn.Linear(
            self.hidden_dim,
            self.latent_dim * self.drift_rank,
        )

        self.tau_head = nn.Linear(
            self.hidden_dim,
            self.latent_dim,
        )

        self.closure_p_head = nn.Linear(
            self.hidden_dim,
            self.latent_dim * self.closure_rank,
        )
        self.closure_q_head = nn.Linear(
            self.hidden_dim,
            self.latent_dim * self.closure_rank,
        )

        common_dim = self.hidden_dim * 5 + self.latent_dim

        self.event_gate_head = nn.Linear(common_dim, 1)
        self.update_gate_head = nn.Linear(
            common_dim,
            self.latent_dim,
        )
        self.output_gate_head = nn.Linear(
            common_dim,
            self.c_out,
        )

        self.output_head = nn.Sequential(
            nn.Linear(common_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(
                self.hidden_dim * 2,
                self.block_size * self.c_out,
            ),
        )

        # 新模块初始时只产生极小残差。
        # 小非零初始化：
        # 初始 correction 仍接近零，
        # 但张量递归分支从第一步即可获得梯度。
        nn.init.normal_(
            self.output_head[-1].weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(
            self.output_head[-1].bias
        )

        # sigmoid(-2.944) ≈ 0.05
        self.residual_scale_raw = nn.Parameter(
            torch.tensor(-1.386)
        )

        self.bypass = env_bool(
            "CPTRD_BYPASS",
            False,
        )

        self.max_residual_scale = env_float(
            "CPTRD_MAX_RESIDUAL_SCALE",
            0.02,
        )

        self.max_correction_rms = env_float(
            "CPTRD_MAX_CORRECTION_RMS",
            0.005,
        )

        self.debug_every = env_int(
            "CPTRD_DEBUG_EVERY",
            200,
        )

        self._forward_calls = 0
        self._last_auxiliary_loss = None
        self._last_diagnostics = {}

        self._last_base_pred = None
        self._last_correction = None
        self._last_event_gate = None

    def _context(self, x_enc: torch.Tensor) -> torch.Tensor:
        mean = x_enc.mean(dim=1)
        std = x_enc.std(dim=1, unbiased=False)
        last = x_enc[:, -1, :]
        slope = x_enc[:, -1, :] - x_enc[:, 0, :]

        return self.context_encoder(
            torch.cat([mean, std, last, slope], dim=-1)
        )

    def _update_state(
        self,
        state_index: int,
        phi: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        own = states[state_index]

        others = torch.cat(
            [
                states[index]
                for index in range(4)
                if index != state_index
            ],
            dim=-1,
        )

        message = torch.tanh(
            self.message_projections[state_index](others)
        )

        gate = torch.sigmoid(
            self.message_gates[state_index](
                torch.cat([own, others], dim=-1)
            )
        )

        cell_input = torch.cat(
            [phi, gate * message],
            dim=-1,
        )

        return self.cells[state_index](cell_input, own)

    def _operators(
        self,
        h_s: torch.Tensor,
        h_d: torch.Tensor,
        h_tau: torch.Tensor,
        h_r: torch.Tensor,
    ):
        batch_size = h_s.shape[0]

        stable_factor = torch.tanh(
            self.stable_head(h_s)
        ).reshape(
            batch_size,
            self.latent_dim,
            self.stable_rank,
        )

        stable_factor = self.stable_scale * stable_factor

        stable = (
            stable_factor
            @ stable_factor.transpose(-1, -2)
            / max(1, self.stable_rank)
        )

        identity = torch.eye(
            self.latent_dim,
            device=h_s.device,
            dtype=h_s.dtype,
        ).unsqueeze(0)

        stable = stable + 1e-4 * identity

        drift_u = torch.tanh(
            self.drift_u_head(h_d)
        ).reshape(
            batch_size,
            self.latent_dim,
            self.drift_rank,
        )

        drift_v = torch.tanh(
            self.drift_v_head(h_d)
        ).reshape(
            batch_size,
            self.latent_dim,
            self.drift_rank,
        )

        raw_drift = (
            drift_u
            @ drift_v.transpose(-1, -2)
            / max(1, self.drift_rank)
        )

        drift = (
            0.5
            * self.drift_scale
            * (raw_drift - raw_drift.transpose(-1, -2))
        )

        tau = self.tau_min + (
            self.tau_max - self.tau_min
        ) * torch.sigmoid(self.tau_head(h_tau))

        inv_sqrt_tau = torch.rsqrt(
            tau.clamp_min(1e-4)
        )

        stable_effective = (
            stable
            * inv_sqrt_tau.unsqueeze(-1)
            * inv_sqrt_tau.unsqueeze(-2)
        )

        closure_p = torch.tanh(
            self.closure_p_head(h_r)
        ).reshape(
            batch_size,
            self.latent_dim,
            self.closure_rank,
        )

        closure_q = torch.tanh(
            self.closure_q_head(h_r)
        ).reshape(
            batch_size,
            self.latent_dim,
            self.closure_rank,
        )

        closure = (
            closure_p
            @ closure_q.transpose(-1, -2)
            / max(1, self.closure_rank)
        )

        return stable_effective, drift, tau, closure

    def _dynamics_step(
        self,
        z: torch.Tensor,
        stable: torch.Tensor,
        drift: torch.Tensor,
        closure: torch.Tensor,
        event_gate: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = z.shape[0]

        identity = torch.eye(
            self.latent_dim,
            device=z.device,
            dtype=z.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        z_column = z.unsqueeze(-1)

        # 隐式耗散步：
        # (I + dt*S) z_s = z
        z_stable = torch.linalg.solve(
            identity + self.dt * stable,
            z_column,
        )

        # Cayley 漂移步：
        # (I - dt/2 D) z_d = (I + dt/2 D) z_s
        z_drift = torch.linalg.solve(
            identity - 0.5 * self.dt * drift,
            (
                identity + 0.5 * self.dt * drift
            ) @ z_stable,
        )

        z_proposed = (
            z_drift
            + self.dt
            * self.closure_scale
            * event_gate
            * (closure @ z_drift)
        )

        return z_proposed.squeeze(-1)

    def forward(
        self,
        base_pred: torch.Tensor,
        x_enc: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon, channels = base_pred.shape

        if horizon != self.pred_len:
            raise RuntimeError(
                f"CPTRD pred_len mismatch: "
                f"expected {self.pred_len}, got {horizon}"
            )

        if channels != self.c_out:
            raise RuntimeError(
                f"CPTRD c_out mismatch: "
                f"expected {self.c_out}, got {channels}"
            )

        if self.bypass:
            zero = base_pred.sum() * 0.0

            self._last_auxiliary_loss = zero
            self._last_diagnostics = {
                "bypass": base_pred.new_tensor(1.0),
                "residual_scale": base_pred.new_tensor(0.0),
                "correction_rms": base_pred.new_tensor(0.0),
            }

            return base_pred

        context = self._context(x_enc)
        z = torch.tanh(self.z_init(context))

        states = tuple(
            torch.tanh(initializer(context))
            for initializer in self.state_initializers
        )

        delta_blocks = []
        tau_values = []
        event_values = []
        closure_norms = []

        for block_index in range(self.n_blocks):
            left = self.block_edges[block_index]
            right = self.block_edges[block_index + 1]

            base_block = base_pred[:, left:right, :]

            block_mean = base_block.mean(dim=1)
            block_std = base_block.std(
                dim=1,
                unbiased=False,
            )
            block_first = base_block[:, 0, :]
            block_last = base_block[:, -1, :]
            block_slope = block_last - block_first

            if base_block.shape[1] > 1:
                block_diff_energy = (
                    base_block[:, 1:, :]
                    - base_block[:, :-1, :]
                ).abs().mean(dim=1)
            else:
                block_diff_energy = torch.zeros_like(
                    block_mean
                )

            block_features = torch.cat(
                [
                    block_mean,
                    block_std,
                    block_first,
                    block_last,
                    block_slope,
                    block_diff_energy,
                ],
                dim=-1,
            )

            phi = torch.tanh(
                context
                + self.block_encoder(block_features)
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

            stable, drift, tau, closure = self._operators(
                h_s,
                h_d,
                h_tau,
                h_r,
            )

            common_before = torch.cat(
                [phi, h_s, h_d, h_tau, h_r, z],
                dim=-1,
            )

            event_gate = torch.sigmoid(
                self.event_gate_head(common_before)
            ).view(batch_size, 1, 1)

            z_proposed = self._dynamics_step(
                z,
                stable,
                drift,
                closure,
                event_gate,
            )

            common_update = torch.cat(
                [phi, h_s, h_d, h_tau, h_r, z_proposed],
                dim=-1,
            )

            update_gate = torch.sigmoid(
                self.update_gate_head(common_update)
            )

            z = (
                (1.0 - update_gate) * z
                + update_gate * z_proposed
            )

            readout = torch.cat(
                [phi, h_s, h_d, h_tau, h_r, z],
                dim=-1,
            )

            block_delta = self.output_head(readout).reshape(
                batch_size,
                self.block_size,
                self.c_out,
            )

            actual_length = right - left
            block_delta = block_delta[:, :actual_length, :]

            output_gate = torch.sigmoid(
                self.output_gate_head(readout)
            ).unsqueeze(1)

            # event_gate 现在不仅调节闭合算子，
            # 也真正控制是否启用该 block 的 correction。
            block_delta = (
                block_delta
                * output_gate
                * event_gate
            )
            delta_blocks.append(block_delta)

            tau_values.append(tau.mean())
            event_values.append(event_gate.mean())
            closure_norms.append(closure.pow(2).mean())

        delta = torch.cat(delta_blocks, dim=1)

        # 有界原始残差。
        delta = torch.tanh(delta)

        residual_scale = (
            self.max_residual_scale
            * torch.sigmoid(
                self.residual_scale_raw
            )
        )

        correction = (
            residual_scale * delta
        )

        correction_rms_per_sample = (
            correction
            .pow(2)
            .mean(dim=(1, 2))
            .add(1e-12)
            .sqrt()
        )

        trust_ratio = torch.clamp(
            self.max_correction_rms
            / correction_rms_per_sample,
            max=1.0,
        )

        correction = (
            correction
            * trust_ratio[:, None, None]
        )

        final_correction_rms = (
            correction
            .pow(2)
            .mean()
            .add(1e-12)
            .sqrt()
        )

        auxiliary_loss = (
            5e-4
            * correction.pow(2).mean()
            + 1e-4
            * torch.stack(
                closure_norms
            ).mean()
        )

        self._last_auxiliary_loss = (
            auxiliary_loss
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
            "residual_scale": (
                residual_scale.detach()
            ),
            "correction_rms": (
                final_correction_rms.detach()
            ),
            "trust_cap_rate": (
                (
                    trust_ratio < 0.999
                )
                .float()
                .mean()
                .detach()
            ),
        }

        self._forward_calls += 1

        if (
            self.training
            and self.debug_every > 0
            and self._forward_calls
            % self.debug_every
            == 0
        ):
            print(
                "[CPTRD_SAFE] "
                f"scale={float(residual_scale):.8f} "
                f"corr_rms="
                f"{float(final_correction_rms):.8f} "
                f"cap_rate="
                f"{float(self._last_diagnostics['trust_cap_rate']):.4f} "
                f"tau="
                f"{float(self._last_diagnostics['tau_mean']):.4f} "
                f"event="
                f"{float(self._last_diagnostics['event_rate']):.4f}"
            )

        self._last_base_pred = base_pred
        self._last_correction = correction
        self._last_event_gate = torch.stack(
            event_values
        ).mean()

        return base_pred + correction

    def supervision_loss(
        self,
        target: torch.Tensor,
    ):
        if (
            self._last_base_pred is None
            or self._last_correction is None
        ):
            zero = self.residual_scale_raw * 0.0
            return zero, {}

        base_pred = self._last_base_pred
        correction = self._last_correction

        target_correction = (
            target - base_pred
        ).detach()

        # 使用信任域尺度归一化，否则 correction 太小，
        # 主 MSE 给它的梯度会非常弱。
        correction_scale = max(
            self.max_correction_rms,
            1e-4,
        )

        normalized_correction = (
            correction / correction_scale
        )

        normalized_target = torch.clamp(
            target_correction
            / correction_scale,
            min=-1.0,
            max=1.0,
        )

        correction_distill_loss = (
            F.smooth_l1_loss(
                normalized_correction,
                normalized_target,
            )
        )

        base_error = base_pred - target
        corrected_error = (
            base_pred
            + correction
            - target
        )

        base_mse = (
            base_error
            .pow(2)
            .mean(dim=(1, 2))
        )

        corrected_mse = (
            corrected_error
            .pow(2)
            .mean(dim=(1, 2))
        )

        base_mae = (
            base_error
            .abs()
            .mean(dim=(1, 2))
        )

        corrected_mae = (
            corrected_error
            .abs()
            .mean(dim=(1, 2))
        )

        relative_delta_mse = (
            corrected_mse - base_mse
        ) / base_mse.detach().clamp_min(1e-6)

        relative_delta_mae = (
            corrected_mae - base_mae
        ) / base_mae.detach().clamp_min(1e-6)

        # 不允许依赖少数大收益样本掩盖大量伤害样本。
        safety_loss = (
            F.relu(relative_delta_mse).mean()
            + 0.25
            * F.relu(relative_delta_mae).mean()
        )

        correction_flat = correction.reshape(
            correction.shape[0],
            -1,
        )

        target_flat = target_correction.reshape(
            target_correction.shape[0],
            -1,
        )

        direction_cosine = (
            F.cosine_similarity(
                correction_flat,
                target_flat,
                dim=1,
                eps=1e-8,
            )
        )

        direction_loss = (
            1.0 - direction_cosine.mean()
        )

        lambda_distill = env_float(
            "CPTRD_LAMBDA_DIRECTION_DISTILL",
            0.05,
        )

        lambda_safety = env_float(
            "CPTRD_LAMBDA_SAMPLE_SAFETY",
            0.20,
        )

        lambda_direction = env_float(
            "CPTRD_LAMBDA_DIRECTION_COS",
            0.02,
        )

        event_target = env_float(
            "CPTRD_EVENT_TARGET_RATE",
            0.20,
        )

        lambda_event_rate = env_float(
            "CPTRD_LAMBDA_EVENT_RATE",
            0.05,
        )

        if self._last_event_gate is None:
            event_rate = correction.new_tensor(
                event_target
            )
        else:
            event_rate = self._last_event_gate

        event_rate_loss = (
            event_rate - event_target
        ).pow(2)

        loss = (
            lambda_distill
            * correction_distill_loss
            + lambda_safety
            * safety_loss
            + lambda_direction
            * direction_loss
            + lambda_event_rate
            * event_rate_loss
        )

        statistics = {
            "direction_cosine": (
                direction_cosine
                .mean()
                .detach()
            ),
            "mse_win_rate": (
                (corrected_mse < base_mse)
                .float()
                .mean()
                .detach()
            ),
            "mae_win_rate": (
                (corrected_mae < base_mae)
                .float()
                .mean()
                .detach()
            ),
            "delta_mse": (
                (
                    corrected_mse
                    - base_mse
                )
                .mean()
                .detach()
            ),
            "delta_mae": (
                (
                    corrected_mae
                    - base_mae
                )
                .mean()
                .detach()
            ),
            "event_rate": (
                event_rate.detach()
            ),
            "event_rate_loss": (
                event_rate_loss.detach()
            ),
            "distill_loss": (
                correction_distill_loss
                .detach()
            ),
            "sample_safety_loss": (
                safety_loss.detach()
            ),
        }

        return loss, statistics

    def auxiliary_loss(self) -> torch.Tensor:
        if self._last_auxiliary_loss is None:
            return self.residual_scale_raw * 0.0
        return self._last_auxiliary_loss

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return self._last_diagnostics


class Model(nn.Module):
    """
    TGCA_Forecaster + CPTRD residual wrapper。
    """

    def __init__(self, configs) -> None:
        super().__init__()

        self.base = TGCA_Forecaster.Model(configs)
        self.cptrd = CoupledParallelTensorDynamics(configs)

        self.freeze_base = env_bool(
            "CPTRD_FREEZE_BASE",
            False,
        )

        if self.freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad = False

        print(
            "[CPTRD_V1] "
            f"blocks={self.cptrd.n_blocks} "
            f"latent={self.cptrd.latent_dim} "
            f"hidden={self.cptrd.hidden_dim} "
            f"freeze_base={int(self.freeze_base)}"
        )

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_base:
            self.base.eval()

        return self

    def forward(
        self,
        x_enc: torch.Tensor,
        *args,
        **kwargs,
    ):
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

        if isinstance(base_output, tuple):
            corrected = self.cptrd(
                base_output[0],
                x_enc,
            )
            return (corrected, *base_output[1:])

        if isinstance(base_output, list):
            corrected = self.cptrd(
                base_output[0],
                x_enc,
            )
            return [corrected, *base_output[1:]]

        return self.cptrd(base_output, x_enc)

    def cptrd_auxiliary_loss(self) -> torch.Tensor:
        return self.cptrd.auxiliary_loss()

    def cptrd_supervision_loss(
        self,
        target: torch.Tensor,
    ):
        return self.cptrd.supervision_loss(
            target
        )

    def cptrd_diagnostics(self) -> dict[str, torch.Tensor]:
        return self.cptrd.diagnostics()

    def load_state_dict(
        self,
        state_dict: Any,
        strict: bool = True,
    ):
        if (
            isinstance(state_dict, dict)
            and "state_dict" in state_dict
            and isinstance(state_dict["state_dict"], dict)
        ):
            state_dict = state_dict["state_dict"]

        cleaned = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            cleaned[key] = value

        has_wrapper_prefix = any(
            key.startswith(("base.", "cptrd."))
            for key in cleaned
        )

        if not has_wrapper_prefix:
            cleaned = {
                f"base.{key}": value
                for key, value in cleaned.items()
            }
            strict = False
            print(
                "[CPTRD_V1] loading plain TGCA checkpoint "
                "into base.*"
            )
        else:
            print("[CPTRD_V1] loading CPTRD checkpoint")

        result = super().load_state_dict(
            cleaned,
            strict=strict,
        )

        if result.missing_keys:
            print(
                "[CPTRD_V1] missing keys:",
                result.missing_keys[:12],
            )

        if result.unexpected_keys:
            print(
                "[CPTRD_V1] unexpected keys:",
                result.unexpected_keys[:12],
            )

        return result
