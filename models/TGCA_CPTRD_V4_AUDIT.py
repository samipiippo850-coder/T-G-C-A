from __future__ import annotations

import atexit
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import TGCA_CPTRD_V4


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1", "true", "yes", "on"
    }


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None:
        return default

    return float(value)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    return int(value)


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        matrix
        + matrix.transpose(-1, -2)
    )


def frobenius_norm(
    matrix: torch.Tensor,
) -> torch.Tensor:
    return torch.linalg.vector_norm(
        matrix,
        dim=(-2, -1),
    )


def regularized_log_condition(
    matrix: torch.Tensor,
    relative_floor: float = 1e-6,
    absolute_floor: float = 1e-12,
) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(
        matrix
    )

    largest = singular_values[..., 0]

    smallest = singular_values[..., -1]

    floor = torch.maximum(
        largest * relative_floor,
        torch.full_like(
            largest,
            absolute_floor,
        ),
    )

    denominator = torch.maximum(
        smallest,
        floor,
    )

    condition = (
        largest
        / denominator.clamp_min(
            absolute_floor
        )
    )

    return torch.log10(
        condition.clamp_min(1.0)
    )


def shrink_covariance(
    trajectory: torch.Tensor,
    shrinkage: float,
    epsilon: float,
) -> torch.Tensor:
    """
    trajectory: [B,K,D]
    """
    batch_size, steps, dimension = (
        trajectory.shape
    )

    centered = (
        trajectory
        - trajectory.mean(
            dim=1,
            keepdim=True,
        )
    )

    covariance = (
        centered.transpose(-1, -2)
        @ centered
    ) / max(steps - 1, 1)

    identity = torch.eye(
        dimension,
        dtype=trajectory.dtype,
        device=trajectory.device,
    ).expand(
        batch_size,
        dimension,
        dimension,
    )

    average_variance = (
        torch.diagonal(
            covariance,
            dim1=-2,
            dim2=-1,
        )
        .mean(dim=-1)
        .clamp_min(epsilon)
    )

    target = (
        average_variance[:, None, None]
        * identity
    )

    return (
        (1.0 - shrinkage)
        * covariance
        + shrinkage
        * target
        + epsilon
        * identity
    )


def covariance_spectrum(
    covariance: torch.Tensor,
    epsilon: float,
):
    eigenvalues, eigenvectors = (
        torch.linalg.eigh(
            symmetrize(covariance)
        )
    )

    eigenvalues = eigenvalues.clamp_min(
        epsilon
    )

    probabilities = (
        eigenvalues
        / eigenvalues.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(epsilon)
    )

    entropy = -(
        probabilities
        * probabilities.clamp_min(
            epsilon
        ).log()
    ).sum(dim=-1)

    effective_rank = torch.exp(entropy)

    square_root = (
        eigenvectors
        * eigenvalues.sqrt().unsqueeze(-2)
    ) @ eigenvectors.transpose(-1, -2)

    inverse_square_root = (
        eigenvectors
        * eigenvalues.rsqrt().unsqueeze(-2)
    ) @ eigenvectors.transpose(-1, -2)

    return (
        eigenvalues,
        eigenvectors,
        effective_rank,
        square_root,
        inverse_square_root,
    )


def operator_metrics(
    prefix: str,
    operator_blocks: torch.Tensor,
    covariance: torch.Tensor,
    covariance_sqrt: torch.Tensor,
    covariance_inv_sqrt: torch.Tensor,
    covariance_eigenvectors: torch.Tensor,
    epsilon: float,
):
    """
    operator_blocks: [B,K,D,D]
    covariance:      [B,D,D]
    """
    mean_operator = operator_blocks.mean(
        dim=1
    )

    block_log_condition = (
        regularized_log_condition(
            operator_blocks
        )
    )

    log_condition = (
        regularized_log_condition(
            mean_operator
        )
    )

    whitened_operator = (
        covariance_inv_sqrt
        @ mean_operator
        @ covariance_sqrt
    )

    whitened_log_condition = (
        regularized_log_condition(
            whitened_operator
        )
    )

    propagated_covariance = (
        mean_operator
        @ covariance
        @ mean_operator.transpose(-1, -2)
    )

    dimension = covariance.shape[-1]

    identity = torch.eye(
        dimension,
        dtype=covariance.dtype,
        device=covariance.device,
    ).expand_as(covariance)

    propagated_covariance = (
        propagated_covariance
        + epsilon * identity
    )

    propagated_log_condition = (
        regularized_log_condition(
            propagated_covariance
        )
    )

    operator_norm = frobenius_norm(
        mean_operator
    )

    covariance_norm = frobenius_norm(
        covariance
    )

    commutator = (
        mean_operator @ covariance
        - covariance @ mean_operator
    )

    commutator_score = (
        frobenius_norm(commutator)
        / (
            operator_norm
            * covariance_norm
            + epsilon
        )
    )

    left_transport = (
        mean_operator @ covariance
    )

    right_transport = (
        covariance
        @ mean_operator.transpose(-1, -2)
    )

    transport_mismatch = (
        frobenius_norm(
            left_transport
            - right_transport
        )
        / (
            frobenius_norm(left_transport)
            + frobenius_norm(right_transport)
            + epsilon
        )
    )

    normality_residual = (
        mean_operator.transpose(-1, -2)
        @ mean_operator
        - mean_operator
        @ mean_operator.transpose(-1, -2)
    )

    non_normality = (
        frobenius_norm(
            normality_residual
        )
        / (
            operator_norm.pow(2)
            + epsilon
        )
    )

    symmetric_eigenvalues = (
        torch.linalg.eigvalsh(
            symmetrize(mean_operator)
        )
    )

    spectral_radius = (
        torch.linalg.eigvals(
            mean_operator
        )
        .abs()
        .amax(dim=-1)
        .real
    )

    top_covariance_direction = (
        covariance_eigenvectors[
            :,
            :,
            -1,
        ]
    )

    top_direction_gain = (
        torch.linalg.vector_norm(
            mean_operator
            @ top_covariance_direction.unsqueeze(-1),
            dim=(-2, -1),
        )
        / (
            operator_norm
            + epsilon
        )
    )

    return {
        f"{prefix}_logcond":
            log_condition,

        f"{prefix}_block_logcond_mean":
            block_log_condition.mean(dim=1),

        f"{prefix}_block_logcond_max":
            block_log_condition.amax(dim=1),

        f"{prefix}_whitened_logcond":
            whitened_log_condition,

        f"{prefix}_prop_cov_logcond":
            propagated_log_condition,

        f"{prefix}_norm":
            operator_norm,

        f"{prefix}_commutator":
            commutator_score,

        f"{prefix}_transport_mismatch":
            transport_mismatch,

        f"{prefix}_non_normality":
            non_normality,

        f"{prefix}_sym_min_eig":
            symmetric_eigenvalues[..., 0],

        f"{prefix}_sym_max_eig":
            symmetric_eigenvalues[..., -1],

        f"{prefix}_spectral_radius":
            spectral_radius,

        f"{prefix}_topcov_gain":
            top_direction_gain,
    }



def hill_tail_index(
    values: torch.Tensor,
    fraction: float,
    minimum_count: int,
    epsilon: float,
) -> torch.Tensor:
    """
    对每个样本的正谱值计算 Hill 型尾指数。

    只作为重尾诊断量，不作为严格幂律显著性检验。
    """
    flattened = (
        values
        .reshape(values.shape[0], -1)
        .abs()
        .clamp_min(epsilon)
    )

    sorted_values = torch.sort(
        flattened,
        dim=-1,
        descending=True,
    ).values

    total = sorted_values.shape[-1]

    if total < 3:
        return torch.full(
            (sorted_values.shape[0],),
            float("nan"),
            dtype=values.dtype,
            device=values.device,
        )

    tail_count = max(
        minimum_count,
        int(round(total * fraction)),
    )

    tail_count = min(
        max(tail_count, 2),
        total - 1,
    )

    tail = sorted_values[
        :,
        :tail_count,
    ]

    threshold = sorted_values[
        :,
        tail_count:tail_count + 1,
    ].clamp_min(epsilon)

    log_excess = torch.log(
        (
            tail
            / threshold
        ).clamp_min(1.0)
    )

    denominator = log_excess.sum(
        dim=-1
    )

    estimate = (
        float(tail_count)
        / denominator.clamp_min(
            epsilon
        )
    )

    return torch.clamp(
        estimate,
        min=0.0,
        max=1e6,
    )


def spectral_extreme_metrics(
    prefix: str,
    operator_blocks: torch.Tensor,
    covariance: torch.Tensor,
    covariance_eigenvalues: torch.Tensor,
    covariance_eigenvectors: torch.Tensor,
    epsilon: float,
):
    """
    operator_blocks: [B,K,D,D]
    covariance:      [B,D,D]
    """
    tail_fraction = env_float(
        "CPTRD_SPECTRAL_TAIL_FRACTION",
        0.10,
    )

    tail_minimum = env_int(
        "CPTRD_SPECTRAL_TAIL_MIN",
        8,
    )

    rank_floor = env_float(
        "CPTRD_SPECTRAL_RANK_FLOOR",
        1e-4,
    )

    mean_operator = operator_blocks.mean(
        dim=1
    )

    (
        left_vectors,
        singular_values,
        right_vectors_h,
    ) = torch.linalg.svd(
        mean_operator,
        full_matrices=False,
    )

    del left_vectors

    block_singular_values = (
        torch.linalg.svdvals(
            operator_blocks
        )
    )

    dimension = singular_values.shape[-1]

    sigma_max = singular_values[..., 0]

    sigma_second = (
        singular_values[..., 1]
        if dimension > 1
        else sigma_max
    )

    sigma_median = singular_values.median(
        dim=-1
    ).values

    relative_threshold = (
        sigma_max.unsqueeze(-1)
        * rank_floor
    )

    retained = (
        singular_values
        >= relative_threshold
    )

    infinity = torch.full_like(
        singular_values,
        float("inf"),
    )

    sigma_min_nonzero = torch.where(
        retained,
        singular_values,
        infinity,
    ).amin(dim=-1)

    sigma_min_nonzero = torch.where(
        torch.isfinite(
            sigma_min_nonzero
        ),
        sigma_min_nonzero,
        sigma_max * rank_floor,
    )

    pseudo_rank = retained.sum(
        dim=-1
    ).to(singular_values.dtype)

    spectral_energy = (
        singular_values.pow(2)
    )

    total_energy = spectral_energy.sum(
        dim=-1
    ).clamp_min(epsilon)

    top_energy = (
        spectral_energy[..., 0]
        / total_energy
    )

    top_two_energy = (
        spectral_energy[
            ...,
            :min(2, dimension),
        ].sum(dim=-1)
        / total_energy
    )

    stable_rank = (
        total_energy
        / sigma_max.pow(2).clamp_min(
            epsilon
        )
    )

    spectral_probability = (
        spectral_energy
        / total_energy.unsqueeze(-1)
    )

    spectral_entropy = -(
        spectral_probability
        * spectral_probability
        .clamp_min(epsilon)
        .log()
    ).sum(dim=-1)

    spectral_effective_rank = torch.exp(
        spectral_entropy
    )

    upper_spike_ratio = (
        sigma_max
        / sigma_second.clamp_min(
            epsilon
        )
    )

    upper_bulk_ratio = (
        sigma_max
        / sigma_median.clamp_min(
            epsilon
        )
    )

    lower_collapse_ratio = (
        sigma_median
        / sigma_min_nonzero.clamp_min(
            epsilon
        )
    )

    block_top = (
        block_singular_values[..., 0]
    )

    block_second = (
        block_singular_values[..., 1]
        if dimension > 1
        else block_top
    )

    block_upper_spike = (
        block_top
        / block_second.clamp_min(
            epsilon
        )
    )

    pooled_spectrum = (
        block_singular_values
        .reshape(
            block_singular_values.shape[0],
            -1,
        )
    )

    pooled_sorted = torch.sort(
        pooled_spectrum,
        dim=-1,
        descending=True,
    ).values

    pooled_count = pooled_sorted.shape[-1]

    top_count = max(
        1,
        int(round(
            pooled_count
            * tail_fraction
        )),
    )

    top_count = min(
        top_count,
        pooled_count,
    )

    pooled_energy = (
        pooled_sorted.pow(2)
    )

    tail_energy_share = (
        pooled_energy[
            :,
            :top_count,
        ].sum(dim=-1)
        / pooled_energy.sum(
            dim=-1
        ).clamp_min(epsilon)
    )

    tail_threshold = pooled_sorted[
        :,
        top_count - 1,
    ]

    max_over_tail_threshold = (
        pooled_sorted[:, 0]
        / tail_threshold.clamp_min(
            epsilon
        )
    )

    tail_alpha = hill_tail_index(
        block_singular_values,
        fraction=tail_fraction,
        minimum_count=tail_minimum,
        epsilon=epsilon,
    )

    top_right_direction = (
        right_vectors_h[
            ...,
            0,
            :,
        ]
    )

    covariance_top_direction = (
        covariance_eigenvectors[
            ...,
            -1,
        ]
    )

    spike_alignment = (
        (
            top_right_direction
            * covariance_top_direction
        )
        .sum(dim=-1)
        .abs()
        .pow(2)
    )

    covariance_top_eigenvalue = (
        covariance_eigenvalues[
            ...,
            -1,
        ]
    )

    excited_spike = (
        sigma_max.pow(2)
        * spike_alignment
        * covariance_top_eigenvalue
    )

    propagated_covariance = (
        mean_operator
        @ covariance
        @ mean_operator.transpose(
            -1,
            -2,
        )
    )

    propagated_eigenvalues = (
        torch.linalg.eigvalsh(
            symmetrize(
                propagated_covariance
            )
        )
        .clamp_min(epsilon)
    )

    propagated_total = (
        propagated_eigenvalues.sum(
            dim=-1
        ).clamp_min(epsilon)
    )

    propagated_top_energy = (
        propagated_eigenvalues[
            ...,
            -1,
        ]
        / propagated_total
    )

    propagated_spike_ratio = (
        propagated_eigenvalues[
            ...,
            -1,
        ]
        / propagated_eigenvalues[
            ...,
            -2,
        ].clamp_min(epsilon)
        if dimension > 1
        else torch.ones_like(
            sigma_max
        )
    )

    return {
        f"{prefix}_sigma_max":
            sigma_max,

        f"{prefix}_sigma_second":
            sigma_second,

        f"{prefix}_sigma_median":
            sigma_median,

        f"{prefix}_sigma_min_nonzero":
            sigma_min_nonzero,

        f"{prefix}_pseudo_rank":
            pseudo_rank,

        f"{prefix}_upper_spike_ratio":
            upper_spike_ratio,

        f"{prefix}_upper_edge_index":
            torch.log10(
                upper_bulk_ratio
                .clamp_min(1.0)
            ),

        f"{prefix}_lower_edge_index":
            torch.log10(
                lower_collapse_ratio
                .clamp_min(1.0)
            ),

        f"{prefix}_top_energy":
            top_energy,

        f"{prefix}_top2_energy":
            top_two_energy,

        f"{prefix}_stable_rank":
            stable_rank,

        f"{prefix}_spectral_entropy":
            spectral_entropy,

        f"{prefix}_spectral_effective_rank":
            spectral_effective_rank,

        f"{prefix}_block_spike_mean":
            block_upper_spike.mean(
                dim=1
            ),

        f"{prefix}_block_spike_max":
            block_upper_spike.amax(
                dim=1
            ),

        f"{prefix}_tail_alpha_hill":
            tail_alpha,

        f"{prefix}_tail_energy_share":
            tail_energy_share,

        f"{prefix}_max_over_tail_threshold":
            max_over_tail_threshold,

        f"{prefix}_cov_alignment":
            spike_alignment,

        f"{prefix}_excited_spike":
            excited_spike,

        f"{prefix}_prop_top_energy":
            propagated_top_energy,

        f"{prefix}_prop_spike_ratio":
            propagated_spike_ratio,
    }


def finite_time_growth_metrics(
    effective_operator: torch.Tensor,
    dt: float,
    epsilon: float,
):
    """
    计算 T = I + dt*A 的有限时间传播增益。
    """
    steps = [1, 2, 4, 8, 16]

    batch_size, dimension, _ = (
        effective_operator.shape
    )

    identity = torch.eye(
        dimension,
        dtype=effective_operator.dtype,
        device=effective_operator.device,
    ).expand(
        batch_size,
        dimension,
        dimension,
    )

    transition = (
        identity
        + dt * effective_operator
    )

    spectral_radius = (
        torch.linalg.eigvals(
            transition
        )
        .abs()
        .amax(dim=-1)
        .real
    )

    power = identity

    gains = []
    excesses = []
    output = {
        "transition_exact_spectral_radius":
            spectral_radius,
    }

    maximum_step = max(steps)

    for current_step in range(
        1,
        maximum_step + 1,
    ):
        power = power @ transition

        if current_step not in steps:
            continue

        gain = torch.linalg.svdvals(
            power
        )[..., 0]

        gain = torch.nan_to_num(
            gain,
            nan=1e12,
            posinf=1e12,
            neginf=0.0,
        )

        asymptotic_scale = (
            spectral_radius
            .clamp_min(1e-8)
            .pow(current_step)
        )

        excess = (
            gain
            / asymptotic_scale.clamp_min(
                epsilon
            )
        )

        gains.append(gain)
        excesses.append(excess)

        output[
            f"transient_gain_k{current_step}"
        ] = gain

        output[
            f"transient_log_gain_k{current_step}"
        ] = torch.log10(
            gain.clamp_min(epsilon)
        )

        output[
            f"transient_excess_k{current_step}"
        ] = excess

    gain_stack = torch.stack(
        gains,
        dim=-1,
    )

    excess_stack = torch.stack(
        excesses,
        dim=-1,
    )

    maximum_gain, maximum_index = (
        gain_stack.max(dim=-1)
    )

    maximum_excess = (
        excess_stack.max(dim=-1).values
    )

    step_tensor = torch.tensor(
        steps,
        dtype=effective_operator.dtype,
        device=effective_operator.device,
    )

    output["transient_gain_max"] = (
        maximum_gain
    )

    output["transient_gain_argmax_k"] = (
        step_tensor[maximum_index]
    )

    output["transient_excess_max"] = (
        maximum_excess
    )

    return output




def transition_metrics(
    latent_states: torch.Tensor,
    effective_operator: torch.Tensor,
    dt: float,
    shrinkage: float,
    epsilon: float,
):
    """
    latent_states: [B,K,D]
    effective_operator: [B,D,D]
    """
    previous = latent_states[:, :-1, :]

    following = latent_states[:, 1:, :]

    previous_centered = (
        previous
        - previous.mean(
            dim=1,
            keepdim=True,
        )
    )

    following_centered = (
        following
        - following.mean(
            dim=1,
            keepdim=True,
        )
    )

    batch_size, steps, dimension = (
        previous_centered.shape
    )

    identity = torch.eye(
        dimension,
        dtype=latent_states.dtype,
        device=latent_states.device,
    ).expand(
        batch_size,
        dimension,
        dimension,
    )

    ridge = env_float(
        "CPTRD_COND_TRANSITION_RIDGE",
        1e-4,
    )

    xtx = (
        previous_centered.transpose(-1, -2)
        @ previous_centered
    ) / max(steps - 1, 1)

    xty = (
        previous_centered.transpose(-1, -2)
        @ following_centered
    ) / max(steps - 1, 1)

    regression = torch.linalg.solve(
        xtx + ridge * identity,
        xty,
    )

    transition = regression.transpose(
        -1,
        -2,
    )

    predicted_following = (
        previous_centered
        @ regression
    )

    innovation = (
        following_centered
        - predicted_following
    )

    covariance_previous = shrink_covariance(
        previous,
        shrinkage,
        epsilon,
    )

    covariance_following = shrink_covariance(
        following,
        shrinkage,
        epsilon,
    )

    covariance_innovation = shrink_covariance(
        innovation,
        shrinkage,
        epsilon,
    )

    discrete_residual = (
        covariance_following
        - transition
        @ covariance_previous
        @ transition.transpose(-1, -2)
        - covariance_innovation
    )

    discrete_denominator = (
        frobenius_norm(
            covariance_following
        )
        + frobenius_norm(
            transition
            @ covariance_previous
            @ transition.transpose(-1, -2)
        )
        + frobenius_norm(
            covariance_innovation
        )
        + epsilon
    )

    discrete_lyapunov_residual = (
        frobenius_norm(
            discrete_residual
        )
        / discrete_denominator
    )

    predicted_transition = (
        identity
        + dt * effective_operator
    )

    operator_transition_mismatch = (
        frobenius_norm(
            transition
            - predicted_transition
        )
        / (
            frobenius_norm(transition)
            + frobenius_norm(
                predicted_transition
            )
            + epsilon
        )
    )

    transition_spectral_radius = (
        torch.linalg.eigvals(
            transition
        )
        .abs()
        .amax(dim=-1)
        .real
    )

    return {
        "transition_logcond":
            regularized_log_condition(
                transition
            ),

        "transition_spectral_radius":
            transition_spectral_radius,

        "discrete_lyapunov_residual":
            discrete_lyapunov_residual,

        "operator_transition_mismatch":
            operator_transition_mismatch,

        "innovation_cov_logcond":
            regularized_log_condition(
                covariance_innovation
            ),
    }


class AuditSignedProxyDynamics(
    TGCA_CPTRD_V4.SignedProxyDynamics
):
    def __init__(self, configs) -> None:
        super().__init__(configs)

        self.audit_enabled = env_bool(
            "CPTRD_COND_AUDIT",
            False,
        )

        self.audit_path = Path(
            os.getenv(
                "CPTRD_COND_AUDIT_PATH",
                "/root/autodl-tmp/"
                "cptrd_conditioning_audit/"
                "conditioning_audit.npz",
            )
        )

        self.shrinkage = env_float(
            "CPTRD_COND_SHRINKAGE",
            0.05,
        )

        self.audit_epsilon = env_float(
            "CPTRD_COND_EPSILON",
            1e-8,
        )

        self.flush_every = env_int(
            "CPTRD_COND_FLUSH_EVERY",
            128,
        )

        self.audit_records = defaultdict(list)
        self.audit_sample_count = 0

        self._audit_stable = []
        self._audit_drift = []
        self._audit_closure = []
        self._audit_relax = []
        self._audit_latent = []
        self._audit_event = []

        if self.audit_enabled:
            atexit.register(
                self.flush_audit
            )

            print(
                "[CPTRD_COND_AUDIT] "
                f"path={self.audit_path} "
                f"shrinkage={self.shrinkage}"
            )

    def _tau_to_operator(
        self,
        tau: torch.Tensor,
        dimension: int,
    ) -> torch.Tensor:
        batch_size = tau.shape[0]

        if (
            tau.ndim >= 3
            and tau.shape[-1] == dimension
            and tau.shape[-2] == dimension
        ):
            return tau

        flattened = tau.reshape(
            batch_size,
            -1,
        )

        if flattened.shape[-1] == 1:
            diagonal = flattened.repeat(
                1,
                dimension,
            )

        elif flattened.shape[-1] == dimension:
            diagonal = flattened

        else:
            diagonal = (
                flattened.mean(
                    dim=-1,
                    keepdim=True,
                )
                .repeat(
                    1,
                    dimension,
                )
            )

        relaxation_rate = (
            1.0
            / diagonal.abs().clamp_min(
                1e-6
            )
        )

        return torch.diag_embed(
            relaxation_rate
        )

    def _operators(
        self,
        h_s: torch.Tensor,
        h_d: torch.Tensor,
        h_tau: torch.Tensor,
        h_r: torch.Tensor,
    ):
        stable, drift, tau, closure = (
            super()._operators(
                h_s,
                h_d,
                h_tau,
                h_r,
            )
        )

        if self.audit_enabled:
            dimension = stable.shape[-1]

            self._audit_stable.append(
                stable.detach()
            )

            self._audit_drift.append(
                drift.detach()
            )

            self._audit_closure.append(
                closure.detach()
            )

            self._audit_relax.append(
                self._tau_to_operator(
                    tau.detach(),
                    dimension,
                )
            )

        return stable, drift, tau, closure

    def _dynamics_step(
        self,
        z: torch.Tensor,
        stable: torch.Tensor,
        drift: torch.Tensor,
        closure: torch.Tensor,
        event_gate: torch.Tensor,
        *args,
        **kwargs,
    ):
        if self.audit_enabled:
            self._audit_latent.append(
                z.detach()
            )

            self._audit_event.append(
                event_gate.detach()
            )

        return super()._dynamics_step(
            z,
            stable,
            drift,
            closure,
            event_gate,
            *args,
            **kwargs,
        )

    def _reset_audit_capture(self) -> None:
        self._audit_stable = []
        self._audit_drift = []
        self._audit_closure = []
        self._audit_relax = []
        self._audit_latent = []
        self._audit_event = []

    def append_audit(
        self,
        metrics: dict[str, torch.Tensor],
    ) -> None:
        batch_size = None

        for name, value in metrics.items():
            value = (
                value.detach()
                .cpu()
                .to(torch.float64)
                .numpy()
            )

            value = np.asarray(value).reshape(-1)

            if batch_size is None:
                batch_size = len(value)

            self.audit_records[name].append(
                value
            )

        if batch_size is None:
            return

        self.audit_sample_count += batch_size

        if (
            self.flush_every > 0
            and self.audit_sample_count
            % self.flush_every
            < batch_size
        ):
            self.flush_audit()

    def flush_audit(self) -> None:
        if (
            not self.audit_enabled
            or not self.audit_records
        ):
            return

        self.audit_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = {
            name: np.concatenate(values)
            for name, values
            in self.audit_records.items()
        }

        length = min(
            len(value)
            for value in output.values()
        )

        output = {
            name: value[:length]
            for name, value
            in output.items()
        }

        output["sample_index"] = np.arange(
            length,
            dtype=np.int64,
        )

        temporary = self.audit_path.with_name(
            self.audit_path.stem
            + ".tmp.npz"
        )

        np.savez_compressed(
            temporary,
            **output,
        )

        os.replace(
            temporary,
            self.audit_path,
        )

        print(
            "[CPTRD_COND_AUDIT] "
            f"saved={self.audit_path} "
            f"n={length} "
            f"features={len(output)-1}"
        )

    def compute_audit(self) -> None:
        if not self.audit_enabled:
            return

        if not self._audit_latent:
            raise RuntimeError(
                "No latent states captured."
            )

        dtype = torch.float64

        stable = torch.stack(
            self._audit_stable,
            dim=1,
        ).to(dtype)

        drift = torch.stack(
            self._audit_drift,
            dim=1,
        ).to(dtype)

        closure = torch.stack(
            self._audit_closure,
            dim=1,
        ).to(dtype)

        relaxation = torch.stack(
            self._audit_relax,
            dim=1,
        ).to(dtype)

        latent = torch.stack(
            self._audit_latent,
            dim=1,
        ).to(dtype)

        event = torch.stack(
            self._audit_event,
            dim=1,
        ).to(dtype)

        covariance = shrink_covariance(
            latent,
            self.shrinkage,
            self.audit_epsilon,
        )

        (
            covariance_eigenvalues,
            covariance_eigenvectors,
            covariance_effective_rank,
            covariance_sqrt,
            covariance_inv_sqrt,
        ) = covariance_spectrum(
            covariance,
            self.audit_epsilon,
        )

        event_mean = event.mean(dim=1)

        effective = (
            stable
            + drift
            + event * closure
            - relaxation
        )

        metrics = {
            "latent_cov_logcond":
                regularized_log_condition(
                    covariance
                ),

            "latent_cov_effective_rank":
                covariance_effective_rank,

            "latent_cov_min_eig":
                covariance_eigenvalues[..., 0],

            "latent_cov_max_eig":
                covariance_eigenvalues[..., -1],

            "latent_cov_trace":
                covariance_eigenvalues.sum(
                    dim=-1
                ),

            "event_rate":
                event_mean.reshape(
                    event_mean.shape[0],
                    -1,
                ).mean(dim=-1),
        }

        covariance_total = (
            covariance_eigenvalues
            .sum(dim=-1)
            .clamp_min(
                self.audit_epsilon
            )
        )

        covariance_top = (
            covariance_eigenvalues[
                ...,
                -1,
            ]
        )

        covariance_second = (
            covariance_eigenvalues[
                ...,
                -2,
            ]
        )

        covariance_median = (
            covariance_eigenvalues
            .median(dim=-1)
            .values
        )

        metrics.update({
            "cov_top_energy":
                covariance_top
                / covariance_total,

            "cov_top2_energy":
                covariance_eigenvalues[
                    ...,
                    -2:,
                ].sum(dim=-1)
                / covariance_total,

            "cov_spike_ratio":
                covariance_top
                / covariance_second
                .clamp_min(
                    self.audit_epsilon
                ),

            "cov_upper_edge_index":
                torch.log10(
                    (
                        covariance_top
                        / covariance_median
                        .clamp_min(
                            self.audit_epsilon
                        )
                    ).clamp_min(1.0)
                ),

            "cov_lower_edge_index":
                torch.log10(
                    (
                        covariance_median
                        / covariance_eigenvalues[
                            ...,
                            0,
                        ].clamp_min(
                            self.audit_epsilon
                        )
                    ).clamp_min(1.0)
                ),

            "cov_spectral_entropy":
                torch.log(
                    covariance_effective_rank
                    .clamp_min(
                        self.audit_epsilon
                    )
                ),
        })

        operator_groups = {
            "stable": stable,
            "drift": drift,
            "closure": closure,
            "relax": relaxation,
            "effective": effective,
        }

        for prefix, operators in (
            operator_groups.items()
        ):
            metrics.update(
                operator_metrics(
                    prefix,
                    operators,
                    covariance,
                    covariance_sqrt,
                    covariance_inv_sqrt,
                    covariance_eigenvectors,
                    self.audit_epsilon,
                )
            )

            metrics.update(
                spectral_extreme_metrics(
                    prefix,
                    operators,
                    covariance,
                    covariance_eigenvalues,
                    covariance_eigenvectors,
                    self.audit_epsilon,
                )
            )

        metrics[
            "drift_minus_stable_logcond"
        ] = (
            metrics["drift_logcond"]
            - metrics["stable_logcond"]
        )

        metrics[
            "drift_minus_stable_whitened_logcond"
        ] = (
            metrics["drift_whitened_logcond"]
            - metrics[
                "stable_whitened_logcond"
            ]
        )

        metrics[
            "drift_to_stable_norm_ratio"
        ] = (
            metrics["drift_norm"]
            / (
                metrics["stable_norm"]
                + self.audit_epsilon
            )
        )

        metrics[
            "closure_to_stable_norm_ratio"
        ] = (
            metrics["closure_norm"]
            / (
                metrics["stable_norm"]
                + self.audit_epsilon
            )
        )

        effective_mean = effective.mean(
            dim=1
        )

        metrics.update(
            finite_time_growth_metrics(
                effective_mean,
                float(self.dt),
                self.audit_epsilon,
            )
        )

        metrics.update(
            transition_metrics(
                latent,
                effective_mean,
                float(self.dt),
                self.shrinkage,
                self.audit_epsilon,
            )
        )

        self.append_audit(metrics)

    def forward(
        self,
        base_pred: torch.Tensor,
        x_enc: torch.Tensor,
        proxy_state: torch.Tensor,
    ) -> torch.Tensor:
        self._reset_audit_capture()

        output = super().forward(
            base_pred,
            x_enc,
            proxy_state,
        )

        if self.audit_enabled:
            with torch.no_grad():
                self.compute_audit()

        return output


class Model(TGCA_CPTRD_V4.Model):
    def __init__(self, configs) -> None:
        super().__init__(configs)

        self.cptrd = AuditSignedProxyDynamics(
            configs
        )

        print(
            "[CPTRD_V4_AUDIT] "
            "read-only conditioning audit enabled"
        )
