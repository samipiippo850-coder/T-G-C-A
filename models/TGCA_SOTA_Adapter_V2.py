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

        self.direction_blocks = min(
            env_int(
                "TGCA_ADAPTER_DIRECTION_BLOCKS",
                24,
            ),
            self.pred_len,
        )

        self.max_signed_gate = env_float(
            "TGCA_ADAPTER_MAX_SIGNED_GATE",
            1.0,
        )

        # Optional conditional residual gain. The zero-initialized head
        # preserves the original V2 path at initialization while allowing
        # different correction strengths per channel and horizon block.
        self.conditional_gain_enabled = env_bool(
            "TGCA_ADAPTER_CONDITIONAL_GAIN",
            False,
        )
        self.gain_hidden = env_int(
            "TGCA_ADAPTER_GAIN_HIDDEN",
            32,
        )
        self.gain_max_delta = env_float(
            "TGCA_ADAPTER_GAIN_MAX_DELTA",
            2.0,
        )
        if self.conditional_gain_enabled:
            self.conditional_gain_head = nn.Sequential(
                nn.Linear(8, self.gain_hidden),
                nn.LayerNorm(self.gain_hidden),
                nn.GELU(),
                nn.Linear(
                    self.gain_hidden,
                    self.direction_blocks,
                ),
            )
            nn.init.zeros_(
                self.conditional_gain_head[-1].weight
            )
            nn.init.zeros_(
                self.conditional_gain_head[-1].bias
            )

        # Parameter-free residual attention driven by candidate energy and
        # spectral temperature. Existing checkpoints remain loadable.
        self.energy_temp_enabled = env_bool(
            "TGCA_ADAPTER_ENERGY_TEMP",
            False,
        )
        self.energy_temp_learnable = env_bool(
            "TGCA_ADAPTER_LEARNABLE_ENERGY_TEMP",
            False,
        )
        self.layer_attention_enabled = env_bool(
            "TGCA_ADAPTER_LAYER_ATTENTION",
            False,
        )
        self.layer_attention_gain = env_float(
            "TGCA_ADAPTER_LAYER_ATTENTION_GAIN",
            0.25,
        )
        self.direction_attention_enabled = env_bool(
            "TGCA_DIRECTION_CONSISTENCY_ATTENTION",
            False,
        )
        self.direction_attention_gain = env_float(
            "TGCA_DIRECTION_ATTENTION_GAIN",
            0.25,
        )
        self.error_aligned_router_enabled = env_bool(
            "TGCA_ERROR_ALIGNED_ROUTER",
            False,
        )
        self.cov_thermo_enabled = env_bool("TGCA_COV_THERMO_ROUTER", False)
        self.cov_thermo_temperature = env_float("TGCA_COV_THERMO_TEMPERATURE", 0.20)
        self.cov_thermo_gain = env_float("TGCA_COV_THERMO_GAIN", 0.20)
        self.cov_hard_matrix_enabled = env_bool("TGCA_COV_HARD_MATRIX", False)
        self.cov_hard_matrix_gain = env_float("TGCA_COV_HARD_MATRIX_GAIN", 0.25)
        self.cov_hard_matrix_cap = env_float("TGCA_COV_HARD_MATRIX_CAP", 0.06)
        self.cov_hard_gate_mode = os.environ.get("TGCA_COV_HARD_GATE_MODE", "risk").strip().lower()
        self.cov_hard_low_power_hi = env_float("TGCA_COV_HARD_LOW_POWER_HI", 0.06)
        self.cov_hard_low_power_lo = env_float("TGCA_COV_HARD_LOW_POWER_LO", 0.0)
        self.cov_hard_low_free_hi = env_float("TGCA_COV_HARD_LOW_FREE_HI", 0.006)
        self.cov_hard_low_free_lo = env_float("TGCA_COV_HARD_LOW_FREE_LO", 0.0)
        self.cov_hard_cond_delta_min = env_float("TGCA_COV_HARD_COND_DELTA_MIN", -0.05)
        self.cov_hard_rank_drop_min = env_float("TGCA_COV_HARD_RANK_DROP_MIN", -0.02)
        self.cov_hard_brownian_delta_min = env_float("TGCA_COV_HARD_BROWNIAN_DELTA_MIN", -1e9)
        self.cov_hard_global_kurtosis_min = env_float("TGCA_COV_HARD_GLOBAL_KURTOSIS_MIN", -1e9)
        self.cov_hard_local_kurtosis_min = env_float("TGCA_COV_HARD_LOCAL_KURTOSIS_MIN", -1e9)
        self.cov_hard_tau_delta_max = env_float("TGCA_COV_HARD_TAU_DELTA_MAX", 1e9)
        self.cov_hard_shift_guard_enabled = env_bool("TGCA_COV_HARD_SHIFT_GUARD", False)
        self.cov_hard_guard_power_max = env_float("TGCA_COV_HARD_GUARD_POWER_MAX", 1e9)
        self.cov_hard_guard_free_max = env_float("TGCA_COV_HARD_GUARD_FREE_MAX", 1e9)
        self.cov_hard_guard_brownian_min = env_float("TGCA_COV_HARD_GUARD_BROWNIAN_MIN", -1e9)
        self.cov_hard_guard_brownian_max = env_float("TGCA_COV_HARD_GUARD_BROWNIAN_MAX", 1e9)
        self.cov_hard_guard_global_kurtosis_min = env_float("TGCA_COV_HARD_GUARD_GLOBAL_KURTOSIS_MIN", -1e9)
        self.cov_hard_guard_global_kurtosis_max = env_float("TGCA_COV_HARD_GUARD_GLOBAL_KURTOSIS_MAX", 1e9)
        self.cov_hard_guard_local_kurtosis_min = env_float("TGCA_COV_HARD_GUARD_LOCAL_KURTOSIS_MIN", -1e9)
        self.cov_hard_guard_local_kurtosis_max = env_float("TGCA_COV_HARD_GUARD_LOCAL_KURTOSIS_MAX", 1e9)
        self.cov_hard_guard_scale = env_float("TGCA_COV_HARD_GUARD_SCALE", 0.005)
        self.cov_hard_phase_guard_enabled = env_bool("TGCA_COV_HARD_PHASE_GUARD", False)
        self.cov_hard_phase_power_center = env_float("TGCA_COV_HARD_PHASE_POWER_CENTER", 0.095)
        self.cov_hard_phase_power_width = env_float("TGCA_COV_HARD_PHASE_POWER_WIDTH", 0.055)
        self.cov_hard_phase_free_center = env_float("TGCA_COV_HARD_PHASE_FREE_CENTER", 0.006)
        self.cov_hard_phase_free_width = env_float("TGCA_COV_HARD_PHASE_FREE_WIDTH", 0.006)
        self.cov_hard_phase_strength = env_float("TGCA_COV_HARD_PHASE_STRENGTH", 0.50)
        self.cov_hard_extra_risk_scale = env_float("TGCA_COV_HARD_EXTRA_RISK_SCALE", 1.0)
        self.cov_hard_easy_suppressor_enabled = env_bool("TGCA_COV_HARD_EASY_SUPPRESSOR", False)
        self.cov_hard_easy_rank_min = env_float("TGCA_COV_HARD_EASY_RANK_MIN", 0.4978)
        self.cov_hard_easy_kurtosis_max = env_float("TGCA_COV_HARD_EASY_KURTOSIS_MAX", 3.4804)
        self.cov_hard_easy_scale = env_float("TGCA_COV_HARD_EASY_SCALE", 0.0)
        self.cov_hard_easy_guard_scale = env_float("TGCA_COV_HARD_EASY_GUARD_SCALE", 0.03)
        self.cov_hard_disagreement_suppressor_enabled = env_bool("TGCA_COV_HARD_DISAGREEMENT_SUPPRESSOR", False)
        self.cov_hard_disagreement_min = env_float("TGCA_COV_HARD_DISAGREEMENT_MIN", 0.0)
        self.cov_hard_route_entropy_max = env_float("TGCA_COV_HARD_ROUTE_ENTROPY_MAX", 1e9)
        self.cov_hard_correction_rms_min = env_float("TGCA_COV_HARD_CORRECTION_RMS_MIN", 0.0)
        self.cov_hard_disagreement_scale = env_float("TGCA_COV_HARD_DISAGREEMENT_SCALE", 0.0)
        self.cov_hard_disagreement_guard_scale = env_float("TGCA_COV_HARD_DISAGREEMENT_GUARD_SCALE", 0.01)
        self.cov_hard_shape_agreement_enabled = env_bool("TGCA_COV_HARD_SHAPE_AGREEMENT", False)
        self.cov_hard_shape_scale = env_float("TGCA_COV_HARD_SHAPE_SCALE", 0.25)
        self.cov_hard_shape_min_corr = env_float("TGCA_COV_HARD_SHAPE_MIN_CORR", 0.0)
        self.cov_hard_shape_pool = int(env_float("TGCA_COV_HARD_SHAPE_POOL", 45.0))
        self.cov_hard_shape_start_frac = env_float("TGCA_COV_HARD_SHAPE_START_FRAC", 0.0)
        self.cov_hard_benefit_tree_enabled = env_bool("TGCA_COV_HARD_BENEFIT_TREE", False)
        self.cov_hard_benefit_brownian_split = env_float("TGCA_COV_HARD_BENEFIT_BROWNIAN_SPLIT", -0.0015)
        self.cov_hard_benefit_local_kurtosis_split = env_float("TGCA_COV_HARD_BENEFIT_LOCAL_KURTOSIS_SPLIT", 3.17)
        self.cov_hard_benefit_reject_scale = env_float("TGCA_COV_HARD_BENEFIT_REJECT_SCALE", 0.0)
        # Optional target-free reliability selector for the extra hidden-pathology
        # route. It is default-off so historical checkpoints retain their exact
        # behavior. It only controls the extra correction, not the transferred
        # baseline route or the ordinary thermo correction.
        self.cov_hard_reliability_gate_enabled = env_bool("TGCA_COV_HARD_RELIABILITY_GATE", False)
        self.cov_hard_reliability_local_kurtosis_min = env_float(
            "TGCA_COV_HARD_RELIABILITY_LOCAL_KURTOSIS_MIN", 4.10
        )
        self.cov_hard_reliability_scale = env_float("TGCA_COV_HARD_RELIABILITY_SCALE", 0.0)
        self.cov_hard_gate_scale = env_float("TGCA_COV_HARD_GATE_SCALE", 0.03)
        self.cov_hard_mae_safe_enabled = env_bool("TGCA_COV_HARD_MAE_SAFE", False)
        self.cov_hard_block_channel_shrink_spec = os.environ.get("TGCA_COV_HARD_BLOCK_CHANNEL_SHRINK", "").strip()
        self.cov_hard_block_channel_shrink_factor = env_float("TGCA_COV_HARD_BLOCK_CHANNEL_SHRINK_FACTOR", 0.0)
        self.cov_hard_block_channel_count = int(env_float("TGCA_COV_HARD_BLOCK_CHANNEL_COUNT", 16.0))
        self.cov_hard_block_channel_free_gate_enabled = env_bool(
            "TGCA_COV_HARD_BLOCK_CHANNEL_FREE_GATE", False
        )
        self.cov_hard_block_channel_free_gate_block = int(
            env_float("TGCA_COV_HARD_BLOCK_CHANNEL_FREE_BLOCK", 15.0)
        )
        self.cov_hard_block_channel_free_gate_channel = int(
            env_float("TGCA_COV_HARD_BLOCK_CHANNEL_FREE_CHANNEL", 2.0)
        )
        self.cov_hard_block_channel_free_max = env_float(
            "TGCA_COV_HARD_BLOCK_CHANNEL_FREE_MAX", 1e9
        )
        self.cov_hard_high_risk_suppressor_enabled = env_bool(
            "TGCA_COV_HARD_HIGH_RISK_SUPPRESSOR", False
        )
        self.cov_hard_high_risk_threshold = env_float(
            "TGCA_COV_HARD_HIGH_RISK_THRESHOLD", 1e9
        )
        self.cov_hard_high_risk_scale = env_float(
            "TGCA_COV_HARD_HIGH_RISK_SCALE", 0.0
        )
        self.cov_hard_l1_ratio_cap = env_float("TGCA_COV_HARD_L1_RATIO_CAP", 1.0)
        self.cov_hard_sign_conflict_scale = env_float("TGCA_COV_HARD_SIGN_CONFLICT_SCALE", 1.0)
        self.cov_hard_small_residual_ratio = env_float("TGCA_COV_HARD_SMALL_RESIDUAL_RATIO", 0.75)
        self.cov_hard_zero_mean_update = env_bool("TGCA_COV_HARD_ZERO_MEAN", False)
        self.cov_hard_smooth_kernel = int(env_float("TGCA_COV_HARD_SMOOTH_KERNEL", 1.0))
        self.cov_hard_trend_sign_guard_enabled = env_bool("TGCA_COV_HARD_TREND_SIGN_GUARD", False)
        self.cov_hard_trend_sign_scale = env_float("TGCA_COV_HARD_TREND_SIGN_SCALE", 0.25)
        self.cov_hard_trend_tail_fraction = env_float("TGCA_COV_HARD_TREND_TAIL_FRACTION", 0.25)
        self.cov_hard_trend_strength_min = env_float("TGCA_COV_HARD_TREND_STRENGTH_MIN", 0.10)
        self.cov_hard_trend_start_frac = env_float("TGCA_COV_HARD_TREND_START_FRAC", 0.0)
        self.cov_hard_eigen_phase_guard_enabled = env_bool("TGCA_COV_HARD_EIGEN_PHASE_GUARD", False)
        self.cov_hard_eigen_phase_scale = env_float("TGCA_COV_HARD_EIGEN_PHASE_SCALE", 0.25)
        self.cov_hard_eigen_phase_trend_min = env_float("TGCA_COV_HARD_EIGEN_PHASE_TREND_MIN", 0.05)
        self.cov_hard_eigen_phase_start_frac = env_float("TGCA_COV_HARD_EIGEN_PHASE_START_FRAC", 0.0)
        self.cov_hard_eigen_phase_conditioned = env_bool("TGCA_COV_HARD_EIGEN_PHASE_CONDITIONED", False)
        self.cov_hard_eigen_phase_power_center = env_float("TGCA_COV_HARD_EIGEN_PHASE_POWER_CENTER", 0.095)
        self.cov_hard_eigen_phase_power_width = env_float("TGCA_COV_HARD_EIGEN_PHASE_POWER_WIDTH", 0.055)
        self.cov_hard_eigen_phase_free_center = env_float("TGCA_COV_HARD_EIGEN_PHASE_FREE_CENTER", 0.006)
        self.cov_hard_eigen_phase_free_width = env_float("TGCA_COV_HARD_EIGEN_PHASE_FREE_WIDTH", 0.006)
        self.cov_hard_eigen_phase_condition_floor = env_float("TGCA_COV_HARD_EIGEN_PHASE_CONDITION_FLOOR", 0.0)
        self.cov_hard_blend_enabled = env_bool("TGCA_COV_HARD_BLEND_HEAD", False)
        self.cov_hard_blend_extended_features = env_bool("TGCA_COV_HARD_BLEND_EXTENDED_FEATURES", False)
        self.cov_hard_blend_max_shrink = env_float("TGCA_COV_HARD_BLEND_MAX_SHRINK", 0.75)
        self.cov_hard_blend_init_logit = env_float("TGCA_COV_HARD_BLEND_INIT_LOGIT", -6.0)
        self.cov_cluster_matrices_enabled = env_bool("TGCA_COV_CLUSTER_MATRICES", False)
        self.cov_cluster_static_enabled = env_bool("TGCA_COV_CLUSTER_STATIC", True)
        self.cov_cluster_count = int(env_float("TGCA_COV_CLUSTER_COUNT", 4.0))
        self.cov_cluster_count = max(1, min(self.cov_cluster_count, 8))
        self.cov_tail_fraction = env_float("TGCA_COV_LOCAL_TAIL_FRACTION", 1.0 / 3.0)
        self.cov_shrinkage = env_float("TGCA_COV_SHRINKAGE", 0.05)
        self.cov_tau_min_r2 = env_float("TGCA_COV_TAU_MIN_R2", 0.50)
        self.cov_condition_margin = env_float("TGCA_COV_CONDITION_MARGIN", 0.50)
        self.cov_rank_margin = env_float("TGCA_COV_RANK_MARGIN", 0.05)
        self.cov_free_margin = env_float("TGCA_COV_FREE_MARGIN", 0.02)
        self.cov_power_margin = env_float("TGCA_COV_POWER_MARGIN", 0.10)
        self.cov_power_scale = env_float("TGCA_COV_POWER_SCALE", max(self.cov_power_margin, 1e-3))
        self.cov_power_exp = env_float("TGCA_COV_POWER_EXP", 1.0)
        self.cov_power_bandpass_enabled = env_bool("TGCA_COV_POWER_BANDPASS", False)
        self.cov_power_band_center = env_float("TGCA_COV_POWER_BAND_CENTER", 0.07)
        self.cov_power_band_width = env_float("TGCA_COV_POWER_BAND_WIDTH", 0.04)
        self.cov_power_band_floor = env_float("TGCA_COV_POWER_BAND_FLOOR", 0.15)
        self.cov_power_band_boost = env_float("TGCA_COV_POWER_BAND_BOOST", 0.0)
        self.cov_free_exp = env_float("TGCA_COV_FREE_EXP", 1.0)
        self.cov_tau_lags = (4, 8, 16, 32, 64, 128)
        if self.cov_thermo_enabled:
            self.cov_thermo_head = nn.Sequential(nn.Linear(18, 24), nn.LayerNorm(24), nn.GELU(), nn.Linear(24, self.direction_blocks))
            self.cov_thermo_signed_head = nn.Sequential(nn.Linear(18, 24), nn.LayerNorm(24), nn.GELU(), nn.Linear(24, self.direction_blocks))
            if self.cov_hard_matrix_enabled:
                matrix_out = self.channels * self.channels
                if self.cov_cluster_matrices_enabled:
                    self.cov_cluster_router = nn.Sequential(
                        nn.Linear(18, 24),
                        nn.LayerNorm(24),
                        nn.GELU(),
                        nn.Linear(24, self.cov_cluster_count),
                    )
                    nn.init.zeros_(self.cov_cluster_router[-1].weight)
                    nn.init.zeros_(self.cov_cluster_router[-1].bias)
                    if self.cov_cluster_static_enabled:
                        self.cov_cluster_matrix_bank = nn.Parameter(
                            torch.zeros(self.cov_cluster_count, self.channels, self.channels)
                        )
                    else:
                        matrix_out *= self.cov_cluster_count
                self.cov_hard_matrix_head = nn.Sequential(
                    nn.Linear(18, 32),
                    nn.LayerNorm(32),
                    nn.GELU(),
                    nn.Linear(32, matrix_out),
                )
                nn.init.zeros_(self.cov_hard_matrix_head[-1].weight)
                nn.init.zeros_(self.cov_hard_matrix_head[-1].bias)
                if self.cov_hard_blend_enabled:
                    blend_feature_dim = 23 if self.cov_hard_blend_extended_features else 18
                    self.cov_hard_blend_head = nn.Sequential(
                        nn.Linear(blend_feature_dim, 16),
                        nn.LayerNorm(16),
                        nn.GELU(),
                        nn.Linear(16, 1),
                    )
                    nn.init.zeros_(self.cov_hard_blend_head[-1].weight)
                    nn.init.constant_(self.cov_hard_blend_head[-1].bias, float(self.cov_hard_blend_init_logit))
            nn.init.zeros_(self.cov_thermo_head[-1].weight)
            nn.init.zeros_(self.cov_thermo_head[-1].bias)
            nn.init.zeros_(self.cov_thermo_signed_head[-1].weight)
            nn.init.zeros_(self.cov_thermo_signed_head[-1].bias)
        self.signed_correction_rms_cap = env_float(
            "TGCA_SIGNED_CORRECTION_RMS_CAP",
            0.08,
        )
        if self.error_aligned_router_enabled:
            self.error_router = nn.Sequential(
                nn.Linear(14, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Linear(32, 6),
            )
            self.signed_correction_head = nn.Sequential(
                nn.Linear(14, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Linear(32, 6),
            )
            nn.init.zeros_(self.error_router[-1].weight)
            nn.init.zeros_(self.error_router[-1].bias)
            nn.init.zeros_(self.signed_correction_head[-1].weight)
            nn.init.zeros_(self.signed_correction_head[-1].bias)
        if self.energy_temp_learnable:
            self.energy_threshold = nn.Parameter(torch.tensor(0.07))
            self.energy_width = nn.Parameter(torch.tensor(0.05))
            self.temperature_center = nn.Parameter(torch.tensor(0.40))
            self.energy_temp_gain_param = nn.Parameter(torch.tensor(0.25))
        else:
            self.energy_temp_gain = env_float(
                "TGCA_ADAPTER_ENERGY_TEMP_GAIN",
                0.25,
            )

        self.learnable_attention_enabled = env_bool(
            "TGCA_ADAPTER_LEARNABLE_ATTENTION",
            False,
        )
        self.learnable_attention_gain = env_float(
            "TGCA_ADAPTER_LEARNABLE_ATTENTION_GAIN",
            0.25,
        )
        if self.learnable_attention_enabled:
            self.attention_head = nn.Sequential(
                nn.Linear(8, self.gain_hidden),
                nn.LayerNorm(self.gain_hidden),
                nn.GELU(),
                nn.Linear(self.gain_hidden, self.direction_blocks),
            )
            nn.init.zeros_(self.attention_head[-1].weight)
            nn.init.zeros_(self.attention_head[-1].bias)

        initial_scale = env_float(
            "TGCA_ADAPTER_INIT_SCALE",
            0.20,
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

        # Signed direction gate:
        # [B, C, feature] -> [B, C, direction_blocks]
        #
        # tanh permits positive, negative, or zero correction.
        self.direction_head = nn.Linear(
            8,
            self.direction_blocks,
        )

        self._print_count = 0
        self._last_diagnostics: dict[str, Any] = {}

        self._initialize_residual_to_zero()
        if self.learnable_attention_enabled and env_bool(
            "TGCA_ADAPTER_TRAIN_ATTENTION_ONLY",
            False,
        ):
            for parameter in self.parameters():
                parameter.requires_grad = False
            for parameter in self.attention_head.parameters():
                parameter.requires_grad = True
        if self.error_aligned_router_enabled and env_bool(
            "TGCA_ADAPTER_TRAIN_ROUTER_ONLY",
            False,
        ):
            for parameter in self.parameters():
                parameter.requires_grad = False
            if self.cov_thermo_enabled:
                modules = [self.cov_thermo_head, self.cov_thermo_signed_head]
                if self.cov_hard_matrix_enabled:
                    modules.append(self.cov_hard_matrix_head)
                    if self.cov_cluster_matrices_enabled:
                        modules.append(self.cov_cluster_router)
            else:
                modules = [self.error_router, self.signed_correction_head]
            if (
                self.cov_thermo_enabled
                and self.cov_hard_matrix_enabled
                and self.cov_hard_blend_enabled
                and env_bool("TGCA_ADAPTER_TRAIN_BLEND_ONLY", False)
            ):
                modules = [self.cov_hard_blend_head]
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True
            if (
                self.cov_thermo_enabled
                and self.cov_hard_matrix_enabled
                and self.cov_cluster_matrices_enabled
                and self.cov_cluster_static_enabled
            ):
                self.cov_cluster_matrix_bank.requires_grad = True

    def _initialize_residual_to_zero(self) -> None:
        """
        ReZero-style initialization.

        Experts retain normal random initialization. Only the
        signed residual direction gate starts at zero, making
        the complete model exactly equal to the canonical base
        at initialization.
        """
        nn.init.zeros_(
            self.direction_head.weight
        )
        nn.init.zeros_(
            self.direction_head.bias
        )

    @staticmethod
    def _safe_standardize(
        value: torch.Tensor,
    ):
        center_mode = os.environ.get('TGCA_EXCHANGE_ADAPTER_CENTER', 'mean')
        if center_mode == 'last':
            mean = value[:, -1:, :]
        elif center_mode == 'mean':
            mean = value.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f'unknown adapter center mode: {center_mode}')

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

        return (
            F.softmax(
                gate_logits,
                dim=-1,
            ),
            gate_features,
        )

    def _covariance_thermo_features(self, normalized_input):
        """Target-free local/global covariance, thermodynamics and ACF tau."""
        x = normalized_input.float()
        batch, length, channels = x.shape
        eps = 1e-5
        eye = torch.eye(channels, device=x.device, dtype=x.dtype).unsqueeze(0)

        def spectrum(values):
            centered = values - values.mean(dim=1, keepdim=True)
            cov = centered.transpose(1, 2) @ centered / max(int(values.shape[1]) - 1, 1)
            diag = torch.diagonal(cov, dim1=-2, dim2=-1)
            cov = cov + (self.cov_shrinkage * diag.mean(dim=-1, keepdim=True) + eps).unsqueeze(-1) * eye
            eig, vec = torch.linalg.eigh(cov)
            return eig.flip(-1).clamp_min(eps), vec.flip(-1)

        global_eig, global_vec = spectrum(x)
        tail = max(24, min(length, int(length * self.cov_tail_fraction)))
        local_eig, local_vec = spectrum(x[:, -tail:, :])

        def spectral_terms(eig):
            total = eig.sum(dim=-1, keepdim=True).clamp_min(eps)
            prob = eig / total
            rank = torch.exp(-(prob * prob.clamp_min(eps).log()).sum(dim=-1)) / max(channels, 1)
            top = eig[..., 0] / total[..., 0]
            cond = torch.log(eig[..., 0] / eig[..., -1])
            temp = max(float(self.cov_thermo_temperature), eps)
            free = -temp * torch.logsumexp(-(eig / total) / temp, dim=-1)
            log_i = torch.arange(1, channels + 1, device=x.device, dtype=x.dtype).log()
            log_i = log_i - log_i.mean()
            log_eig = eig.clamp_min(eps).log()
            log_eig = log_eig - log_eig.mean(dim=-1, keepdim=True)
            power = (log_eig * log_i).sum(dim=-1) / (log_i.pow(2).sum() + eps)
            return cond, rank, top, free, power

        gc, gr, gt, gf, gp = spectral_terms(global_eig)
        lc, lr, lt, lf, lp = spectral_terms(local_eig)
        gpj = global_vec[..., :1] @ global_vec[..., :1].transpose(-1, -2)
        lpj = local_vec[..., :1] @ local_vec[..., :1].transpose(-1, -2)
        cov_shift = (gpj - lpj).pow(2).mean(dim=(-2, -1)).sqrt()
        alignment = (global_vec[..., 0] * local_vec[..., 0]).sum(dim=-1).abs()

        def tau(values):
            y = values - values.mean(dim=1, keepdim=True)
            xx, yy, masks = [], [], []
            for lag in self.cov_tau_lags:
                if lag >= int(values.shape[1]) - 2:
                    continue
                a, b = y[:, :-lag, :], y[:, lag:, :]
                corr = ((a * b).sum(dim=1) / torch.sqrt(a.pow(2).sum(dim=1) * b.pow(2).sum(dim=1) + eps).clamp_min(eps)).mean(dim=-1).abs()
                xx.append(torch.full_like(corr, float(lag)))
                yy.append(corr.clamp_min(eps))
                masks.append((corr >= 0.05) & (corr <= 0.98))
            if len(xx) < 2:
                return torch.zeros(batch, device=x.device), torch.zeros(batch, device=x.device)
            lag_x, log_y, mask = torch.stack(xx, -1), torch.log(torch.stack(yy, -1)), torch.stack(masks, -1)
            count = mask.sum(-1)
            xm = (lag_x * mask).sum(-1) / count.clamp_min(1)
            ym = (log_y * mask).sum(-1) / count.clamp_min(1)
            dx, dy = lag_x - xm.unsqueeze(-1), log_y - ym.unsqueeze(-1)
            slope = (dx * dy * mask).sum(-1) / ((dx.pow(2) * mask).sum(-1) + eps)
            fitted = (ym - slope * xm).unsqueeze(-1) + slope.unsqueeze(-1) * lag_x
            rss = ((log_y - fitted).pow(2) * mask).sum(-1)
            tss = ((log_y - ym.unsqueeze(-1)).pow(2) * mask).sum(-1)
            r2 = 1.0 - rss / tss.clamp_min(eps)
            ok = (count >= 2) & (slope < -eps) & (r2 >= self.cov_tau_min_r2)
            return torch.where(ok, (-1.0 / slope).clamp(0.0, 4096.0).log1p(), torch.zeros_like(slope)), ok.to(x.dtype)

        gtau, gv = tau(x)
        ltau, lv = tau(x[:, -tail:, :])
        dtau = (ltau - gtau) * gv * lv

        def moment_terms(values):
            centered = values - values.mean(dim=1, keepdim=True)
            var = centered.pow(2).mean(dim=1).clamp_min(eps)
            fourth = centered.pow(4).mean(dim=1)
            return (fourth / var.pow(2)).mean(dim=-1)

        def diffusion_trace(values):
            if int(values.shape[1]) < 2:
                return torch.zeros(batch, device=x.device, dtype=x.dtype)
            dx = values[:, 1:, :] - values[:, :-1, :]
            return 0.5 * dx.pow(2).mean(dim=(1, 2))

        global_kurtosis = moment_terms(x)
        local_kurtosis = moment_terms(x[:, -tail:, :])
        global_diffusion = diffusion_trace(x)
        local_diffusion = diffusion_trace(x[:, -tail:, :])
        brownian_energy_delta = local_diffusion - global_diffusion

        features = torch.stack([gc, lc, gr, lr, gt, lt, cov_shift, alignment, gf, lf, gp, lp, gtau, ltau, dtau, gv, lv, gv * lv], -1).to(normalized_input.dtype)
        cond_margin = max(float(self.cov_condition_margin), eps)
        rank_margin = max(float(self.cov_rank_margin), eps)
        free_margin = max(float(self.cov_free_margin), eps)
        power_margin = max(float(self.cov_power_margin), eps)
        power_scale = max(float(self.cov_power_scale), eps)
        cond_risk = torch.sigmoid((lc - gc - cond_margin) / cond_margin)
        rank_risk = torch.sigmoid((gr - lr - rank_margin) / rank_margin)
        free_risk = torch.sigmoid((torch.abs(lf - gf) - free_margin) / free_margin)
        power_delta_abs = torch.abs(lp - gp)
        power_risk = torch.sigmoid((power_delta_abs - power_margin) / power_scale)
        if self.cov_power_bandpass_enabled:
            band_center = max(float(self.cov_power_band_center), eps)
            band_width = max(float(self.cov_power_band_width), eps)
            band = torch.exp(-0.5 * ((power_delta_abs - band_center) / band_width).pow(2))
            band_boost = max(float(self.cov_power_band_boost), 0.0)
            if band_boost > 0.0:
                power_risk = power_risk * (1.0 + band_boost * band)
            else:
                band_floor = min(max(float(self.cov_power_band_floor), 0.0), 1.0)
                band = band_floor + (1.0 - band_floor) * band
                power_risk = power_risk * band
        free_exp = max(float(self.cov_free_exp), eps)
        power_exp = max(float(self.cov_power_exp), eps)
        free_risk = free_risk.clamp_min(eps).pow(free_exp)
        power_risk = power_risk.clamp_min(eps).pow(power_exp)
        tau_valid = (gv * lv).clamp(0.0, 1.0)
        tau_risk_value = torch.sigmoid((dtau - 0.20) / 0.20)
        tau_risk = torch.where(tau_valid > 0.5, tau_risk_value, torch.ones_like(tau_risk_value))
        risk = cond_risk * rank_risk * free_risk * power_risk * tau_risk
        diagnostics = {"tau_global_log": gtau, "tau_local_log": ltau, "tau_delta_log": dtau, "cov_global_log_condition": gc, "cov_local_log_condition": lc, "cov_global_effective_rank": gr, "cov_local_effective_rank": lr, "free_energy_global": gf, "free_energy_local": lf, "power_slope_global": gp, "power_slope_local": lp, "power_slope_delta_abs": power_delta_abs, "thermo_cov_risk": risk, "brownian_energy_delta": brownian_energy_delta, "global_diffusion_trace": global_diffusion, "local_diffusion_trace": local_diffusion, "global_kurtosis_mean": global_kurtosis, "local_kurtosis_mean": local_kurtosis, "cov_global_top_eigenvector": global_vec[..., 0], "cov_local_top_eigenvector": local_vec[..., 0]}
        return features, risk, diagnostics

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
        (
            gate_weights,
            gate_features,
        ) = self._moment_gate(
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
        expert_disagreement_rms = torch.sqrt(
            (expert_outputs - mixed_normalized.unsqueeze(-1)).pow(2).mean(dim=(1, 2, 3)) + 1e-12
        ).view(-1, 1)
        route_entropy = (
            -(gate_weights.clamp_min(1e-8) * gate_weights.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=-1)
        ).view(-1, 1)

        adapter_scale = (
            self.max_scale
            * torch.sigmoid(
                self.scale_logit
            )
        )

        # [B, C, K]
        signed_blocks = (
            self.max_signed_gate
            * torch.tanh(
                self.direction_head(
                    gate_features
                )
            )
        )

        # [B, C, K] -> [B, C, H] -> [B, H, C]
        signed_curve = F.interpolate(
            signed_blocks,
            size=self.pred_len,
            mode="linear",
            align_corners=False,
        ).transpose(
            1,
            2,
        )

        # Bound the basis before applying the signed gate.
        residual_basis = torch.tanh(
            mixed_normalized
        )

        residual = (
            adapter_scale
            * signed_curve
            * residual_basis
            * std
        )

        if self.conditional_gain_enabled:
            gain_blocks = 1.0 + self.gain_max_delta * torch.tanh(
                self.conditional_gain_head(gate_features)
            )
            gain_curve = F.interpolate(
                gain_blocks,
                size=self.pred_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            residual = residual * gain_curve
        else:
            gain_curve = torch.ones(
                (x_enc.size(0), self.pred_len, self.channels),
                device=x_enc.device,
                dtype=x_enc.dtype,
            )

        expert_residuals = (
            adapter_scale
            * signed_curve.unsqueeze(-1)
            * torch.tanh(expert_outputs)
            * std.unsqueeze(-1)
            * gain_curve.unsqueeze(-1)
        )
        gated_expert_residuals = (
            expert_residuals
            * gate_weights.unsqueeze(1)
        )
        layer_attention = torch.ones(
            (x_enc.size(0), 3),
            device=x_enc.device,
            dtype=x_enc.dtype,
        )
        if self.layer_attention_enabled:
            late_start = int(2 * self.pred_len / 3)
            trend_late_energy = gated_expert_residuals[
                :, late_start:, :, 0
            ].pow(2).mean(dim=(1, 2))
            trend_risk = (
                (trend_late_energy - 0.006) / 0.006
            ).clamp(0.0, 1.0)
            layer_attention[:, 0] = (
                1.0 - self.layer_attention_gain * trend_risk
            ).clamp(0.65, 1.0)
            layer_delta = (
                gated_expert_residuals
                * (layer_attention[:, None, None, :] - 1.0)
            ).sum(dim=-1)
            residual = residual + layer_delta
        if self.direction_attention_enabled:
            late_start = int(2 * self.pred_len / 3)
            trend_late = gated_expert_residuals[
                :, late_start:, :, 0
            ]
            spectral_late = gated_expert_residuals[
                :, late_start:, :, 2
            ]
            cosine = (trend_late * spectral_late).sum(dim=(1, 2)) / (
                torch.sqrt(trend_late.pow(2).sum(dim=(1, 2)) + 1e-8)
                * torch.sqrt(spectral_late.pow(2).sum(dim=(1, 2)) + 1e-8)
            ).clamp(-1.0, 1.0)
            trend_late_energy = trend_late.pow(2).mean(dim=(1, 2))
            energy_risk = (
                (trend_late_energy - 0.006) / 0.006
            ).clamp(0.0, 1.0)
            direction_conflict = (-cosine).clamp(0.0, 1.0)
            direction_risk = energy_risk * direction_conflict
            layer_attention[:, 0] = (
                1.0
                - self.direction_attention_gain * direction_risk
            ).clamp(0.65, 1.0)
            layer_delta = (
                gated_expert_residuals
                * (layer_attention[:, None, None, :] - 1.0)
            ).sum(dim=-1)
            residual = residual + layer_delta

        # Per-expert residual-layer diagnostics. These describe where each
        # branch puts energy, how hot its spectrum is, and its direction.
        # They are diagnostics only and do not alter the prediction.
        layer_energy = expert_residuals.pow(2).mean(dim=(1, 2))
        layer_direction = expert_residuals.mean(dim=(1, 2))
        layer_positive_rate = (
            expert_residuals > 0.0
        ).float().mean(dim=(1, 2))
        layer_segment_energy = []
        for seg in range(3):
            st = int(seg * self.pred_len / 3)
            ed = int((seg + 1) * self.pred_len / 3)
            layer_segment_energy.append(
                expert_residuals[:, st:ed].pow(2).mean(dim=(1, 2))
            )
        layer_segment_energy = torch.stack(
            layer_segment_energy, dim=1
        )
        layer_spectrum = torch.fft.rfft(
            expert_residuals.float(), dim=1
        )
        layer_power = (
            layer_spectrum.real.pow(2)
            + layer_spectrum.imag.pow(2)
        ).mean(dim=2)
        layer_prob = layer_power / (
            layer_power.sum(dim=1, keepdim=True) + 1e-8
        )
        layer_temperature = -(
            layer_prob.clamp_min(1e-8)
            * layer_prob.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(
            max(int(layer_prob.shape[1]), 2)
        )

        # High candidate energy or high spectral temperature receives less
        # trust. Temperature is normalized spectral entropy of the lookback.
        if self.energy_temp_enabled:
            candidate_rms = torch.sqrt(
                residual.pow(2).mean(dim=(1, 2), keepdim=True)
                + 1e-12
            )
            if self.energy_temp_learnable:
                energy_threshold = self.energy_threshold
                energy_width = self.energy_width.abs().clamp_min(0.01)
                temperature_center = self.temperature_center
                energy_temp_gain = self.energy_temp_gain_param.sigmoid()
            else:
                energy_threshold = 0.07
                energy_width = 0.05
                temperature_center = 0.40
                energy_temp_gain = self.energy_temp_gain
            if self.energy_temp_learnable:
                energy_signal = torch.sigmoid(
                    (candidate_rms - energy_threshold)
                    / energy_width
                )
            else:
                energy_signal = (
                    (candidate_rms - energy_threshold) / energy_width
                ).clamp(0.0, 1.0)
            spectrum = torch.fft.rfft(normalized.float(), dim=1)
            spectral_power = (
                spectrum.real.pow(2) + spectrum.imag.pow(2)
            ).mean(dim=-1)
            spectral_prob = spectral_power / (
                spectral_power.sum(dim=1, keepdim=True) + 1e-8
            )
            spectral_entropy = -(
                spectral_prob.clamp_min(1e-8)
                * spectral_prob.clamp_min(1e-8).log()
            ).sum(dim=1, keepdim=True) / math.log(
                max(int(spectral_prob.shape[1]), 2)
            )
            temperature_signal = (
                spectral_entropy.unsqueeze(-1) - temperature_center
            ).clamp(0.0, 0.60)
            residual_attention = (
                1.0
                - energy_temp_gain
                * (
                    0.80 * energy_signal
                    + (0.20 / 0.60) * temperature_signal
                )
            ).clamp(0.65, 1.05)
            residual = residual * residual_attention
        else:
            residual_attention = torch.ones(
                (x_enc.size(0), 1, 1),
                device=x_enc.device,
                dtype=x_enc.dtype,
            )

        if self.error_aligned_router_enabled:
            channel_layer_energy = expert_residuals.pow(2).mean(dim=1)
            channel_layer_direction = expert_residuals.mean(dim=1)
            router_features = torch.cat(
                [
                    gate_features,
                    torch.log1p(channel_layer_energy),
                    channel_layer_direction,
                ],
                dim=-1,
            )
            router_blocks = 1.0 + 0.25 * torch.tanh(
                self.error_router(router_features)
            )
            router_curve = F.interpolate(
                router_blocks,
                size=self.pred_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            signed_blocks = 0.10 * torch.tanh(
                self.signed_correction_head(router_features)
            )
            signed_correction = F.interpolate(
                signed_blocks,
                size=self.pred_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2) * std
            signed_rms = torch.sqrt(
                signed_correction.pow(2).mean(dim=(1, 2), keepdim=True)
                + 1e-12
            )
            signed_cap = (
                self.signed_correction_rms_cap / signed_rms
            ).clamp(max=1.0)
            signed_correction = signed_correction * signed_cap
            residual = residual * router_curve + signed_correction
        else:
            router_curve = torch.ones(
                (x_enc.size(0), self.pred_len, self.channels),
                device=x_enc.device,
                dtype=x_enc.dtype,
            )
            signed_correction = torch.zeros_like(residual)

        thermo_diagnostics = {}
        hard_update_rms = torch.zeros((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
        hard_shape_agreement = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
        thermo_risk = torch.zeros((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
        if self.cov_thermo_enabled:
            thermo_features, thermo_risk, thermo_diagnostics = self._covariance_thermo_features(normalized)
            thermo_blocks = 1.0 + self.cov_thermo_gain * thermo_risk * torch.tanh(self.cov_thermo_head(thermo_features))
            thermo_curve = F.interpolate(thermo_blocks.unsqueeze(1), size=self.pred_len, mode="linear", align_corners=False).transpose(1, 2)
            residual = residual * thermo_curve
            thermo_signed_blocks = 0.02 * thermo_risk * torch.tanh(self.cov_thermo_signed_head(thermo_features))
            thermo_signed = F.interpolate(thermo_signed_blocks.unsqueeze(1), size=self.pred_len, mode="linear", align_corners=False).transpose(1, 2) * std
            thermo_signed_rms = torch.sqrt(thermo_signed.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-12)
            residual_rms_before_thermo = torch.sqrt(residual.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-12)
            thermo_cap = torch.minimum(torch.full_like(thermo_signed_rms, 0.02), 0.15 * residual_rms_before_thermo) / thermo_signed_rms.clamp_min(1e-12)
            signed_correction = signed_correction + (thermo_signed * thermo_cap.clamp(max=1.0)).clamp(-0.02, 0.02)
            hard_matrix_norm = torch.zeros((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_hard_blend_shrink = torch.zeros((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_hard_blend_mean = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_hard_shift_guard = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_hard_phase_guard = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_hard_trend_sign_guard = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_hard_eigen_phase_guard = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            hard_shape_agreement = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_cluster_weights = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            if self.cov_hard_matrix_enabled:
                if self.cov_cluster_matrices_enabled:
                    cov_cluster_weights = torch.softmax(self.cov_cluster_router(thermo_features), dim=-1)
                    if self.cov_cluster_static_enabled:
                        raw_matrix = self.cov_cluster_matrix_bank.to(device=x_enc.device, dtype=x_enc.dtype).unsqueeze(0)
                        raw_matrix = 0.5 * (raw_matrix + raw_matrix.transpose(2, 3))
                        eye = torch.eye(self.channels, device=x_enc.device, dtype=x_enc.dtype).view(1, 1, self.channels, self.channels)
                        raw_matrix = raw_matrix - eye * raw_matrix.diagonal(dim1=2, dim2=3).unsqueeze(2)
                        raw_matrix = torch.tanh(raw_matrix)
                        matrix_norm = torch.linalg.matrix_norm(raw_matrix, ord="fro", dim=(2, 3), keepdim=False).expand(x_enc.size(0), -1)
                        raw_matrix = raw_matrix / matrix_norm[:1].view(1, self.cov_cluster_count, 1, 1).clamp_min(1e-6)
                    else:
                        raw_matrix = self.cov_hard_matrix_head(thermo_features).view(
                            -1, self.cov_cluster_count, self.channels, self.channels
                        )
                        raw_matrix = 0.5 * (raw_matrix + raw_matrix.transpose(2, 3))
                        eye = torch.eye(self.channels, device=x_enc.device, dtype=x_enc.dtype).view(1, 1, self.channels, self.channels)
                        raw_matrix = raw_matrix - eye * raw_matrix.diagonal(dim1=2, dim2=3).unsqueeze(2)
                        raw_matrix = torch.tanh(raw_matrix)
                        matrix_norm = torch.linalg.matrix_norm(raw_matrix, ord="fro", dim=(2, 3), keepdim=False)
                        raw_matrix = raw_matrix / matrix_norm.view(-1, self.cov_cluster_count, 1, 1).clamp_min(1e-6)
                    hard_matrix = torch.sum(raw_matrix * cov_cluster_weights.view(-1, self.cov_cluster_count, 1, 1), dim=1)
                    hard_matrix_norm = torch.sum(matrix_norm * cov_cluster_weights, dim=1, keepdim=True)
                else:
                    hard_matrix = self.cov_hard_matrix_head(thermo_features).view(-1, self.channels, self.channels)
                    hard_matrix = 0.5 * (hard_matrix + hard_matrix.transpose(1, 2))
                    eye = torch.eye(self.channels, device=x_enc.device, dtype=x_enc.dtype).unsqueeze(0)
                    hard_matrix = hard_matrix - eye * hard_matrix.diagonal(dim1=1, dim2=2).unsqueeze(1)
                    hard_matrix = torch.tanh(hard_matrix)
                    hard_matrix_norm = torch.linalg.matrix_norm(hard_matrix, ord="fro", dim=(1, 2), keepdim=False).unsqueeze(-1)
                    hard_matrix = hard_matrix / hard_matrix_norm.view(-1, 1, 1).clamp_min(1e-6)
                hard_base_risk = thermo_risk
                if self.cov_hard_gate_mode in {"low_spectral", "hidden_pathology", "low_power_free"}:
                    gate_scale = max(float(self.cov_hard_gate_scale), 1e-6)
                    power_delta = thermo_diagnostics.get("power_slope_delta_abs", torch.zeros_like(thermo_risk.view(-1)))
                    free_delta = torch.abs(
                        thermo_diagnostics.get("free_energy_local", torch.zeros_like(power_delta))
                        - thermo_diagnostics.get("free_energy_global", torch.zeros_like(power_delta))
                    )
                    cond_delta = (
                        thermo_diagnostics.get("cov_local_log_condition", torch.zeros_like(power_delta))
                        - thermo_diagnostics.get("cov_global_log_condition", torch.zeros_like(power_delta))
                    )
                    rank_drop = (
                        thermo_diagnostics.get("cov_global_effective_rank", torch.zeros_like(power_delta))
                        - thermo_diagnostics.get("cov_local_effective_rank", torch.zeros_like(power_delta))
                    )
                    low_power_gate = torch.sigmoid((float(self.cov_hard_low_power_hi) - power_delta) / gate_scale)
                    power_lo = max(float(self.cov_hard_low_power_lo), 0.0)
                    if power_lo > 0.0:
                        low_power_gate = low_power_gate * torch.sigmoid((power_delta - power_lo) / gate_scale)
                    low_free_gate = torch.sigmoid((float(self.cov_hard_low_free_hi) - free_delta) / gate_scale)
                    free_lo = max(float(self.cov_hard_low_free_lo), 0.0)
                    if free_lo > 0.0:
                        low_free_gate = low_free_gate * torch.sigmoid((free_delta - free_lo) / gate_scale)
                    cond_gate = torch.sigmoid((cond_delta - float(self.cov_hard_cond_delta_min)) / gate_scale)
                    rank_gate = torch.sigmoid((rank_drop - float(self.cov_hard_rank_drop_min)) / gate_scale)
                    brownian_delta = thermo_diagnostics.get("brownian_energy_delta", torch.zeros_like(power_delta))
                    global_kurtosis = thermo_diagnostics.get("global_kurtosis_mean", torch.zeros_like(power_delta))
                    local_kurtosis = thermo_diagnostics.get("local_kurtosis_mean", torch.zeros_like(power_delta))
                    tau_delta = thermo_diagnostics.get("tau_delta_log", torch.zeros_like(power_delta))
                    brownian_gate = torch.sigmoid((brownian_delta - float(self.cov_hard_brownian_delta_min)) / gate_scale)
                    global_kurtosis_gate = torch.sigmoid((global_kurtosis - float(self.cov_hard_global_kurtosis_min)) / gate_scale)
                    local_kurtosis_gate = torch.sigmoid((local_kurtosis - float(self.cov_hard_local_kurtosis_min)) / gate_scale)
                    tau_gate = torch.sigmoid((float(self.cov_hard_tau_delta_max) - tau_delta) / gate_scale)
                    hidden_gate = (
                        low_power_gate
                        * low_free_gate
                        * cond_gate
                        * rank_gate
                        * brownian_gate
                        * global_kurtosis_gate
                        * local_kurtosis_gate
                        * tau_gate
                    ).view(-1, 1)
                    if self.cov_hard_shift_guard_enabled:
                        guard_scale = max(float(self.cov_hard_guard_scale), 1e-6)
                        guard_power = torch.sigmoid((float(self.cov_hard_guard_power_max) - power_delta) / guard_scale)
                        guard_free = torch.sigmoid((float(self.cov_hard_guard_free_max) - free_delta) / guard_scale)
                        guard_brownian_hi = torch.sigmoid((float(self.cov_hard_guard_brownian_max) - brownian_delta) / guard_scale)
                        guard_brownian_lo = torch.sigmoid((brownian_delta - float(self.cov_hard_guard_brownian_min)) / guard_scale)
                        guard_gk_hi = torch.sigmoid((float(self.cov_hard_guard_global_kurtosis_max) - global_kurtosis) / guard_scale)
                        guard_gk_lo = torch.sigmoid((global_kurtosis - float(self.cov_hard_guard_global_kurtosis_min)) / guard_scale)
                        guard_lk_hi = torch.sigmoid((float(self.cov_hard_guard_local_kurtosis_max) - local_kurtosis) / guard_scale)
                        guard_lk_lo = torch.sigmoid((local_kurtosis - float(self.cov_hard_guard_local_kurtosis_min)) / guard_scale)
                        cov_hard_shift_guard = (
                            guard_power
                            * guard_free
                            * guard_brownian_hi
                            * guard_brownian_lo
                            * guard_gk_hi
                            * guard_gk_lo
                            * guard_lk_hi
                            * guard_lk_lo
                        ).view(-1, 1)
                    if self.cov_hard_gate_mode == "hidden_pathology":
                        extra_hidden_risk = (torch.maximum(thermo_risk, hidden_gate) - thermo_risk).clamp_min(0.0)
                        extra_scale = min(max(float(self.cov_hard_extra_risk_scale), 0.0), 1.0)
                        if self.cov_hard_easy_suppressor_enabled:
                            easy_scale = min(max(float(self.cov_hard_easy_scale), 0.0), 1.0)
                            easy_guard_scale = max(float(self.cov_hard_easy_guard_scale), 1e-6)
                            local_rank = thermo_diagnostics.get("cov_local_effective_rank", torch.zeros_like(power_delta))
                            local_kurtosis = thermo_diagnostics.get("local_kurtosis_mean", torch.zeros_like(power_delta))
                            rank_gate = torch.sigmoid((local_rank - float(self.cov_hard_easy_rank_min)) / easy_guard_scale)
                            kurt_gate = torch.sigmoid((float(self.cov_hard_easy_kurtosis_max) - local_kurtosis) / easy_guard_scale)
                            easy_keep = (rank_gate * kurt_gate).view(-1, 1)
                            easy_factor = easy_scale + (1.0 - easy_scale) * easy_keep
                        else:
                            easy_factor = 1.0
                        if self.cov_hard_benefit_tree_enabled:
                            brownian_delta = thermo_diagnostics.get("brownian_energy_delta", torch.zeros_like(power_delta))
                            local_kurtosis = thermo_diagnostics.get("local_kurtosis_mean", torch.zeros_like(power_delta))
                            reject = (
                                (brownian_delta <= float(self.cov_hard_benefit_brownian_split))
                                & (local_kurtosis > float(self.cov_hard_benefit_local_kurtosis_split))
                            ).view(-1, 1)
                            reject_scale = min(max(float(self.cov_hard_benefit_reject_scale), 0.0), 1.0)
                            benefit_factor = torch.where(
                                reject,
                                torch.full_like(extra_hidden_risk, reject_scale),
                                torch.ones_like(extra_hidden_risk),
                            )
                        else:
                            benefit_factor = 1.0
                        if self.cov_hard_reliability_gate_enabled:
                            # Keep the extra hidden correction only where the
                            # target-free local-tail shape indicates a reliable
                            # correction direction. Fractional scale is supported
                            # for validation probes; 0.0 is the conservative gate.
                            reliability_scale = min(
                                max(float(self.cov_hard_reliability_scale), 0.0), 1.0
                            )
                            local_kurtosis = thermo_diagnostics.get(
                                "local_kurtosis_mean", torch.zeros_like(power_delta)
                            )
                            reliable = (
                                local_kurtosis
                                > float(self.cov_hard_reliability_local_kurtosis_min)
                            ).view(-1, 1)
                            reliability_factor = torch.where(
                                reliable,
                                torch.ones_like(extra_hidden_risk),
                                torch.full_like(extra_hidden_risk, reliability_scale),
                            )
                        else:
                            reliability_factor = 1.0
                        hard_base_risk = thermo_risk + extra_scale * easy_factor * benefit_factor * reliability_factor * cov_hard_shift_guard * extra_hidden_risk
                    else:
                        hard_base_risk = cov_hard_shift_guard * hidden_gate
                    if self.cov_hard_phase_guard_enabled:
                        # Target-free transition guard: validation/test diagnostics show the
                        # middle power-law transition band is where MSE gains often turn
                        # into MAE drift.  We therefore suppress only the extra hard route
                        # energy near that phase boundary, while preserving low/high
                        # spectral regimes where the correction is more stable.
                        power_width = max(float(self.cov_hard_phase_power_width), 1e-6)
                        free_width = max(float(self.cov_hard_phase_free_width), 1e-6)
                        phase_power = torch.exp(
                            -0.5 * ((power_delta - float(self.cov_hard_phase_power_center)) / power_width).pow(2)
                        )
                        phase_free = torch.exp(
                            -0.5 * ((free_delta - float(self.cov_hard_phase_free_center)) / free_width).pow(2)
                        )
                        phase_strength = min(max(float(self.cov_hard_phase_strength), 0.0), 1.0)
                        cov_hard_phase_guard = (1.0 - phase_strength * phase_power * phase_free).clamp(0.0, 1.0).view(-1, 1)
                        if self.cov_hard_gate_mode == "hidden_pathology":
                            extra_risk = (hard_base_risk - thermo_risk).clamp_min(0.0)
                            hard_base_risk = thermo_risk + cov_hard_phase_guard * extra_risk
                        else:
                            hard_base_risk = cov_hard_phase_guard * hard_base_risk
                hard_gate = self.cov_hard_matrix_gain * hard_base_risk.view(-1, 1, 1)
                hard_delta = torch.matmul(residual, hard_matrix) * hard_gate
                hard_delta_rms = torch.sqrt(hard_delta.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-12)
                residual_rms_for_hard = torch.sqrt(residual.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-12)
                hard_cap = torch.minimum(
                    torch.full_like(hard_delta_rms, self.cov_hard_matrix_cap),
                    0.15 * residual_rms_for_hard,
                ) / hard_delta_rms.clamp_min(1e-12)
                hard_update = hard_delta * hard_cap.clamp(max=1.0)
                if self.cov_hard_block_channel_shrink_spec:
                    shrink_factor = min(max(float(self.cov_hard_block_channel_shrink_factor), 0.0), 1.0)
                    block_count = max(1, int(self.cov_hard_block_channel_count))
                    pred_len_actual = int(hard_update.shape[1])
                    for item in self.cov_hard_block_channel_shrink_spec.split(','):
                        item = item.strip()
                        if not item or ':' not in item:
                            continue
                        try:
                            parts = item.split(':')
                            block_idx = int(parts[0])
                            channel_idx = int(parts[1])
                            # Backward compatible syntax is block:channel. An
                            # optional third field allows a local intensity,
                            # which is needed because the same channel can be
                            # helpful at one horizon block and harmful at
                            # another.
                            local_shrink = shrink_factor if len(parts) < 3 else min(
                                max(float(parts[2]), 0.0), 1.0
                            )
                        except (ValueError, IndexError):
                            continue
                        if block_idx < 0 or block_idx >= block_count or channel_idx < 0 or channel_idx >= self.channels:
                            continue
                        start_idx = int(round(pred_len_actual * block_idx / block_count))
                        end_idx = int(round(pred_len_actual * (block_idx + 1) / block_count))
                        if end_idx > start_idx:
                            local_shrink_tensor = hard_update.new_full(
                                (hard_update.shape[0], 1, 1), local_shrink
                            )
                            if (
                                self.cov_hard_block_channel_free_gate_enabled
                                and block_idx == self.cov_hard_block_channel_free_gate_block
                                and channel_idx == self.cov_hard_block_channel_free_gate_channel
                            ):
                                free_local = thermo_diagnostics.get(
                                    "free_energy_local", torch.zeros((hard_update.shape[0],), device=hard_update.device, dtype=hard_update.dtype)
                                )
                                free_global = thermo_diagnostics.get(
                                    "free_energy_global", torch.zeros_like(free_local)
                                )
                                free_delta = torch.abs(free_local - free_global)
                                # The correction is suppressed only in the
                                # low-free-energy-shift regime selected on
                                # validation-dev; outside it, preserve the
                                # original route (factor 1).
                                local_shrink_tensor = torch.where(
                                    (free_delta <= float(self.cov_hard_block_channel_free_max)).view(-1, 1, 1),
                                    local_shrink_tensor,
                                    torch.ones_like(local_shrink_tensor),
                                )
                            hard_update[:, start_idx:end_idx, channel_idx] = hard_update[:, start_idx:end_idx, channel_idx] * local_shrink_tensor
                if self.cov_hard_blend_enabled:
                    blend_max_shrink = min(max(float(self.cov_hard_blend_max_shrink), 0.0), 1.0)
                    blend_features = thermo_features
                    if self.cov_hard_blend_extended_features:
                        extra_blend_features = torch.stack(
                            [
                                thermo_diagnostics.get("brownian_energy_delta", torch.zeros_like(thermo_risk.view(-1))),
                                thermo_diagnostics.get("global_diffusion_trace", torch.zeros_like(thermo_risk.view(-1))),
                                thermo_diagnostics.get("local_diffusion_trace", torch.zeros_like(thermo_risk.view(-1))),
                                thermo_diagnostics.get("global_kurtosis_mean", torch.zeros_like(thermo_risk.view(-1))),
                                thermo_diagnostics.get("local_kurtosis_mean", torch.zeros_like(thermo_risk.view(-1))),
                            ],
                            dim=-1,
                        ).to(thermo_features.dtype)
                        blend_features = torch.cat([thermo_features, extra_blend_features], dim=-1)
                    blend_shrink = blend_max_shrink * torch.sigmoid(self.cov_hard_blend_head(blend_features))
                    cov_hard_blend_shrink = blend_shrink
                    hard_blend = (1.0 - blend_shrink).view(-1, 1, 1)
                    cov_hard_blend_mean = hard_blend.view(-1, 1)
                    hard_update = hard_update * hard_blend
                if self.cov_hard_mae_safe_enabled:
                    if self.cov_hard_zero_mean_update:
                        hard_update = hard_update - hard_update.mean(dim=1, keepdim=True)
                    smooth_kernel = max(1, int(self.cov_hard_smooth_kernel))
                    if smooth_kernel > 1:
                        if smooth_kernel % 2 == 0:
                            smooth_kernel += 1
                        pad = smooth_kernel // 2
                        hard_update = F.avg_pool1d(
                            F.pad(hard_update.transpose(1, 2), (pad, pad), mode="replicate"),
                            kernel_size=smooth_kernel,
                            stride=1,
                        ).transpose(1, 2)
                    conflict_scale = min(max(float(self.cov_hard_sign_conflict_scale), 0.0), 1.0)
                    if conflict_scale < 1.0:
                        residual_abs_mean = residual.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
                        small_residual = residual.abs() < (float(self.cov_hard_small_residual_ratio) * residual_abs_mean)
                        sign_conflict = (hard_update * residual) < 0.0
                        safe_scale = torch.where(
                            small_residual & sign_conflict,
                            torch.full_like(hard_update, conflict_scale),
                            torch.ones_like(hard_update),
                        )
                        hard_update = hard_update * safe_scale
                    l1_ratio_cap = max(float(self.cov_hard_l1_ratio_cap), 0.0)
                    if l1_ratio_cap < 1.0:
                        hard_l1 = hard_update.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
                        residual_l1 = residual.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
                        l1_scale = (l1_ratio_cap * residual_l1 / hard_l1).clamp(max=1.0)
                        hard_update = hard_update * l1_scale
                trend_tail = int(round(float(self.cov_hard_trend_tail_fraction) * normalized.shape[1]))
                trend_tail = max(4, min(int(normalized.shape[1]), trend_tail))
                recent = normalized[:, -trend_tail:, :]
                diff = recent[:, 1:, :] - recent[:, :-1, :]
                trend = diff.mean(dim=1, keepdim=True)
                trend_abs = diff.abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
                trend_strength = trend.abs() / trend_abs
                if self.cov_hard_shape_agreement_enabled:
                    pool = max(1, int(self.cov_hard_shape_pool))
                    if pool > 1:
                        # Downsample to low-frequency shape; area pooling avoids
                        # overreacting to high-frequency noise that mostly affects MSE.
                        hard_shape = F.adaptive_avg_pool1d(hard_update.transpose(1, 2), pool).transpose(1, 2)
                        residual_shape = F.adaptive_avg_pool1d(residual.transpose(1, 2), pool).transpose(1, 2)
                    else:
                        hard_shape = hard_update
                        residual_shape = residual
                    hs = hard_shape - hard_shape.mean(dim=1, keepdim=True)
                    rs = residual_shape - residual_shape.mean(dim=1, keepdim=True)
                    shape_corr = (hs * rs).mean(dim=(1, 2), keepdim=True) / (
                        torch.sqrt(hs.pow(2).mean(dim=(1, 2), keepdim=True) * rs.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-12)
                    )
                    hard_shape_agreement = shape_corr.view(-1, 1)
                    shape_scale = min(max(float(self.cov_hard_shape_scale), 0.0), 1.0)
                    min_corr = float(self.cov_hard_shape_min_corr)
                    shape_guard = torch.where(
                        shape_corr < min_corr,
                        torch.full_like(shape_corr, shape_scale),
                        torch.ones_like(shape_corr),
                    )
                    start_frac = min(max(float(self.cov_hard_shape_start_frac), 0.0), 1.0)
                    if start_frac > 0.0:
                        start_idx = int(round(hard_update.shape[1] * start_frac))
                        if start_idx > 0:
                            shape_time_guard = torch.ones_like(hard_update)
                            shape_time_guard[:, start_idx:, :] = shape_guard
                            hard_update = hard_update * shape_time_guard
                        else:
                            hard_update = hard_update * shape_guard
                    else:
                        hard_update = hard_update * shape_guard
                if self.cov_hard_disagreement_suppressor_enabled:
                    dis_scale = min(max(float(self.cov_hard_disagreement_scale), 0.0), 1.0)
                    dis_guard_scale = max(float(self.cov_hard_disagreement_guard_scale), 1e-6)
                    hard_update_rms_pre = torch.sqrt(hard_update.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-12)
                    disagreement_gate = torch.sigmoid(
                        (expert_disagreement_rms - float(self.cov_hard_disagreement_min)) / dis_guard_scale
                    ).view(-1, 1, 1)
                    entropy_gate = torch.sigmoid(
                        (float(self.cov_hard_route_entropy_max) - route_entropy) / dis_guard_scale
                    ).view(-1, 1, 1)
                    correction_gate = torch.sigmoid(
                        (hard_update_rms_pre.view(-1, 1) - float(self.cov_hard_correction_rms_min)) / dis_guard_scale
                    ).view(-1, 1, 1)
                    need_gate = disagreement_gate * entropy_gate * correction_gate
                    hard_update = hard_update * (dis_scale + (1.0 - dis_scale) * need_gate)
                if self.cov_hard_trend_sign_guard_enabled:
                    # Target-free phase/sign controller.  A local hard update that
                    # opposes a sufficiently persistent recent input drift is a common
                    # source of small absolute-error drift: it may still reduce large
                    # squared errors, but it hurts MAE by adding the wrong local phase.
                    conflict = (hard_update * trend) < 0.0
                    strong_trend = trend_strength > float(self.cov_hard_trend_strength_min)
                    trend_scale = min(max(float(self.cov_hard_trend_sign_scale), 0.0), 1.0)
                    trend_guard = torch.where(
                        conflict & strong_trend,
                        torch.full_like(hard_update, trend_scale),
                        torch.ones_like(hard_update),
                    )
                    start_frac = min(max(float(self.cov_hard_trend_start_frac), 0.0), 1.0)
                    if start_frac > 0.0:
                        start_idx = int(round(hard_update.shape[1] * start_frac))
                        if start_idx > 0:
                            trend_guard[:, :start_idx, :] = 1.0
                    cov_hard_trend_sign_guard = trend_guard.mean(dim=(1, 2), keepdim=True).view(-1, 1)
                    hard_update = hard_update * trend_guard
                if self.cov_hard_eigen_phase_guard_enabled:
                    # Target-free covariance phase controller.  The local top
                    # eigenvector represents the dominant channel-coupled energy
                    # allocation axis.  We align its arbitrary sign with recent drift,
                    # then only shrink the hard-update component projected opposite to
                    # that local eigendirection. Orthogonal correction is preserved.
                    eig = thermo_diagnostics.get("cov_local_top_eigenvector")
                    if eig is not None:
                        eig = eig.to(device=hard_update.device, dtype=hard_update.dtype)
                        eig = eig / torch.linalg.vector_norm(eig, dim=-1, keepdim=True).clamp_min(1e-6)
                        trend_vec = trend.squeeze(1)
                        trend_proj = (trend_vec * eig).sum(dim=-1, keepdim=True)
                        eig = eig * torch.where(trend_proj < 0.0, -torch.ones_like(trend_proj), torch.ones_like(trend_proj))
                        trend_proj_abs = trend_proj.abs()
                        trend_norm = torch.linalg.vector_norm(trend_vec, dim=-1, keepdim=True).clamp_min(1e-6)
                        strong_axis = trend_proj_abs / trend_norm > float(self.cov_hard_eigen_phase_trend_min)
                        proj = (hard_update * eig.view(-1, 1, self.channels)).sum(dim=-1, keepdim=True)
                        eigen_scale = min(max(float(self.cov_hard_eigen_phase_scale), 0.0), 1.0)
                        if self.cov_hard_eigen_phase_conditioned:
                            power_delta = thermo_diagnostics.get("power_slope_delta_abs")
                            free_global = thermo_diagnostics.get("free_energy_global")
                            free_local = thermo_diagnostics.get("free_energy_local")
                            if power_delta is not None and free_global is not None and free_local is not None:
                                power_width = max(float(self.cov_hard_eigen_phase_power_width), 1e-6)
                                free_width = max(float(self.cov_hard_eigen_phase_free_width), 1e-6)
                                free_delta = torch.abs(free_local - free_global)
                                power_phase = torch.exp(
                                    -0.5 * ((power_delta - float(self.cov_hard_eigen_phase_power_center)) / power_width).pow(2)
                                )
                                free_phase = torch.exp(
                                    -0.5 * ((free_delta - float(self.cov_hard_eigen_phase_free_center)) / free_width).pow(2)
                                )
                                floor = min(max(float(self.cov_hard_eigen_phase_condition_floor), 0.0), 1.0)
                                condition = (floor + (1.0 - floor) * power_phase * free_phase).view(-1, 1, 1)
                                eigen_scale_tensor = 1.0 - condition * (1.0 - eigen_scale)
                            else:
                                eigen_scale_tensor = torch.full_like(proj, eigen_scale)
                        else:
                            eigen_scale_tensor = torch.full_like(proj, eigen_scale)
                        conflict = (proj < 0.0) & strong_axis.view(-1, 1, 1)
                        shrink = torch.where(
                            conflict,
                            eigen_scale_tensor.expand_as(proj),
                            torch.ones_like(proj),
                        )
                        start_frac = min(max(float(self.cov_hard_eigen_phase_start_frac), 0.0), 1.0)
                        if start_frac > 0.0:
                            start_idx = int(round(hard_update.shape[1] * start_frac))
                            if start_idx > 0:
                                shrink[:, :start_idx, :] = 1.0
                        hard_update = hard_update - (1.0 - shrink) * proj * eig.view(-1, 1, self.channels)
                        cov_hard_eigen_phase_guard = shrink.mean(dim=(1, 2), keepdim=True).view(-1, 1)
                if self.cov_hard_high_risk_suppressor_enabled:
                    high_risk = thermo_risk.view(-1) >= float(self.cov_hard_high_risk_threshold)
                    high_risk_scale = min(max(float(self.cov_hard_high_risk_scale), 0.0), 1.0)
                    risk_factor = torch.where(
                        high_risk.view(-1, 1, 1),
                        torch.full_like(hard_update, high_risk_scale),
                        torch.ones_like(hard_update),
                    )
                    hard_update = hard_update * risk_factor
                hard_update_rms = torch.sqrt(hard_update.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-12).view(-1, 1)
                residual = residual + hard_update
            thermo_curve = thermo_curve
        else:
            hard_matrix_norm = torch.zeros((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_hard_blend_shrink = torch.zeros((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_hard_blend_mean = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_hard_shift_guard = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_hard_phase_guard = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_hard_trend_sign_guard = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_hard_eigen_phase_guard = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            cov_cluster_weights = torch.ones((x_enc.size(0), 1), device=x_enc.device, dtype=x_enc.dtype)
            thermo_curve = torch.ones_like(residual)

        if self.learnable_attention_enabled:
            attention_blocks = 1.0 + self.learnable_attention_gain * torch.tanh(
                self.attention_head(gate_features)
            )
            attention_curve = F.interpolate(
                attention_blocks,
                size=self.pred_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            attention_rms = torch.sqrt(
                attention_curve.pow(2).mean(dim=(1, 2), keepdim=True)
                + 1e-8
            )
            attention_curve = attention_curve / attention_rms
            residual = residual * attention_curve
        else:
            attention_curve = torch.ones(
                (x_enc.size(0), self.pred_len, self.channels),
                device=x_enc.device,
                dtype=x_enc.dtype,
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

        direction_mean = signed_curve.mean()
        direction_abs_mean = (
            signed_curve.abs().mean()
        )
        direction_positive_rate = (
            signed_curve > 0.0
        ).float().mean()

        self._last_diagnostics = {
            "adapter_scale":
                adapter_scale.detach(),
            "residual_rms":
                final_rms.detach(),
            "expert_disagreement_rms":
                expert_disagreement_rms.detach(),
            "route_entropy":
                route_entropy.detach(),
            "hard_update_rms":
                hard_update_rms.detach(),
            "hard_shape_agreement":
                hard_shape_agreement.detach(),
            "gate_mean":
                gate_mean.detach(),
            "direction_mean":
                direction_mean.detach(),
            "direction_abs_mean":
                direction_abs_mean.detach(),
            "direction_positive_rate":
                direction_positive_rate.detach(),
            "gain_curve":
                gain_curve.detach(),
            "learnable_attention":
                attention_curve.detach(),
            "residual_attention":
                residual_attention.detach(),
            "layer_energy":
                layer_energy.detach(),
            "layer_direction":
                layer_direction.detach(),
            "layer_positive_rate":
                layer_positive_rate.detach(),
            "layer_segment_energy":
                layer_segment_energy.detach(),
            "layer_temperature":
                layer_temperature.detach(),
            "layer_attention":
                layer_attention.detach(),
            "router_curve":
                router_curve.detach(),
            "signed_correction":
                signed_correction.detach(),
            "thermo_curve":
                thermo_curve.detach(),
            "thermo_risk":
                thermo_risk.detach(),
            "cov_hard_matrix_norm":
                hard_matrix_norm.detach(),
            "cov_hard_blend_shrink":
                cov_hard_blend_shrink.detach(),
            "cov_hard_blend_mean":
                cov_hard_blend_mean.detach(),
            "cov_hard_shift_guard":
                cov_hard_shift_guard.detach(),
            "cov_hard_phase_guard":
                cov_hard_phase_guard.detach(),
            "cov_hard_trend_sign_guard":
                cov_hard_trend_sign_guard.detach(),
            "cov_hard_eigen_phase_guard":
                cov_hard_eigen_phase_guard.detach(),
            "cov_cluster_weights":
                cov_cluster_weights.detach(),
        }
        if thermo_diagnostics:
            self._last_diagnostics.update({key: value.detach() for key, value in thermo_diagnostics.items()})

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
                    "direction_mean":
                        float(
                            direction_mean.detach().cpu()
                        ),
                    "direction_abs_mean":
                        float(
                            direction_abs_mean.detach().cpu()
                        ),
                    "direction_positive_rate":
                        float(
                            direction_positive_rate.detach().cpu()
                        ),
                    "max_rms":
                        self.max_rms,
                },
                flush=True,
            )

            self._print_count += 1

        component_export_dir = os.environ.get(
            "TGCA_COMPONENT_EXPORT_DIR",
            "",
        ).strip()

        if component_export_dir and not self.training:
            os.makedirs(
                component_export_dir,
                exist_ok=True,
            )

            part_index = int(
                getattr(
                    self,
                    "_component_export_index",
                    0,
                )
            )

            component_path = os.path.join(
                component_export_dir,
                f"part_{part_index:06d}.pt",
            )

            torch.save(
                {
                    "base_prediction":
                        base_prediction.detach()
                        .cpu().float(),

                    # [B,H,C,3]:
                    # trend, local, spectral
                    "expert_outputs":
                        expert_outputs.detach()
                        .cpu().float(),

                    # [B,C,3]
                    "gate_weights":
                        gate_weights.detach()
                        .cpu().float(),

                    "gate_features":
                        gate_features.detach()
                        .cpu().float(),

                    "mixed_normalized":
                        mixed_normalized.detach()
                        .cpu().float(),

                    "signed_curve":
                        signed_curve.detach()
                        .cpu().float(),

                    "input_std":
                        std.detach()
                        .cpu().float(),

                    "adapter_scale":
                        adapter_scale.detach()
                        .cpu().float(),

                    "cap_factor":
                        cap_factor.detach()
                        .cpu().float(),

                    "residual":
                        residual.detach()
                        .cpu().float(),

                    "max_rms":
                        torch.tensor(
                            float(self.max_rms),
                            dtype=torch.float32,
                        ),
                },
                component_path,
            )

            self._component_export_index = (
                part_index + 1
            )

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

        # ===== TGCA_BASE_SEQ_LEN_ARG_V1 =====
        # CLI --base_seq_len wins; 0/None falls back to the legacy env knob.
        self.base_seq_len = int(
            getattr(
                configs,
                "base_seq_len",
                0,
            )
            or 0
        )

        if self.base_seq_len <= 0:
            self.base_seq_len = env_int(
                "TGCA_ADAPTER_BASE_SEQ_LEN",
                96,
            )
        # ===== TGCA_BASE_SEQ_LEN_ARG_V1_END =====

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

        if self.base_seq_len < 8:
            raise ValueError(
                {
                    "base_seq_len":
                        self.base_seq_len,
                    "reason":
                        "base_seq_len must be >= 8",
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
