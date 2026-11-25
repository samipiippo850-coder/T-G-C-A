# -*- coding: utf-8 -*-
"""
TGCA-Forecaster-L

A correction-native long-horizon forecasting model:
Base Forecast Branch
  -> Residual State Estimator
  -> Correction Operator Bank
  -> Recursive Branch Router
  -> Hard-Projected Trust Region
  -> Final Forecast

Input:
    x_enc: [B, seq_len, C]

Output:
    y_pred: [B, pred_len, C]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TGCAConfig:
    seq_len: int = 96
    pred_len: int = 720
    enc_in: int = 7
    moving_avg: int = 25
    individual: bool = False
    use_revin: bool = True

    d_state: int = 128
    dropout: float = 0.10
    n_ops: int = 6
    n_branches: int = 4
    n_segments: int = 3

    delta_scale: float = 0.20
    alpha_max: float = 0.80
    alpha_fallback: float = 0.05
    alpha_damp_cap: float = 0.20
    alpha_neutral_cap: float = 0.10
    js_limit: float = 0.20
    router_tau: float = 1.0
    eps: float = 1e-6


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        xt = x.transpose(1, 2)
        xt = F.pad(xt, (self.padding, self.padding), mode="replicate")
        out = F.avg_pool1d(xt, kernel_size=self.kernel_size, stride=1)
        return out.transpose(1, 2)


class SeriesDecomp(nn.Module):
    def __init__(self, moving_avg: int):
        super().__init__()
        self.ma = MovingAverage(moving_avg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.ma(x)
        seasonal = x - trend
        return seasonal, trend


class DecompositionLinearBase(nn.Module):
    """
    DLinear-like base branch.
    It only produces y0. TGCA correction is inside the whole model.
    """

    def __init__(self, cfg: TGCAConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.enc_in = cfg.enc_in
        self.individual = cfg.individual
        self.decomp = SeriesDecomp(cfg.moving_avg)

        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)]
            )
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

        self._init_as_mean_forecaster()

    def _init_as_mean_forecaster(self):
        if self.individual:
            for a, b in zip(self.linear_seasonal, self.linear_trend):
                nn.init.constant_(a.weight, 1.0 / self.seq_len)
                nn.init.constant_(b.weight, 1.0 / self.seq_len)
                nn.init.zeros_(a.bias)
                nn.init.zeros_(b.bias)
        else:
            nn.init.constant_(self.linear_seasonal.weight, 1.0 / self.seq_len)
            nn.init.constant_(self.linear_trend.weight, 1.0 / self.seq_len)
            nn.init.zeros_(self.linear_seasonal.bias)
            nn.init.zeros_(self.linear_trend.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(x)
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)

        if self.individual:
            outs = []
            for i in range(self.enc_in):
                yi = self.linear_seasonal[i](seasonal[:, i, :]) + self.linear_trend[i](trend[:, i, :])
                outs.append(yi.unsqueeze(-1))
            y0 = torch.cat(outs, dim=-1)
        else:
            y0 = self.linear_seasonal(seasonal) + self.linear_trend(trend)
            y0 = y0.transpose(1, 2)
        return y0


def spectral_prob(residual: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    spec = torch.fft.rfft(residual, dim=1, norm="ortho")
    energy = (spec.real.pow(2) + spec.imag.pow(2)).mean(dim=-1)
    return (energy + eps) / (energy.sum(dim=-1, keepdim=True) + eps)


def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def js_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, eps) + 0.5 * kl_div(q, m, eps)



def safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Batchwise cosine between two [B,H,C] tensors.
    return: [B,1,1]
    """
    num = (a * b).sum(dim=(1, 2), keepdim=True)
    den = torch.sqrt(a.pow(2).sum(dim=(1, 2), keepdim=True) + eps) * torch.sqrt(b.pow(2).sum(dim=(1, 2), keepdim=True) + eps)
    return num / den.clamp_min(eps)


def segment_energy_ratio(r_base: torch.Tensor, r_final: torch.Tensor, n_segments: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """
    Segment-wise log energy ratio: log(E_final/E_base), shape [B,n_segments].
    Positive means the segment got worse.
    """
    b, h, c = r_base.shape
    outs = []
    for i in range(n_segments):
        st = int(i * h / n_segments)
        ed = int((i + 1) * h / n_segments)
        eb = r_base[:, st:ed, :].pow(2).mean(dim=(1, 2))
        ef = r_final[:, st:ed, :].pow(2).mean(dim=(1, 2))
        outs.append(torch.log((ef + eps) / (eb + eps)))
    return torch.stack(outs, dim=-1)


def make_tgca_branch_oracle(
    r_base: torch.Tensor,
    delta_raw: torch.Tensor,
    alpha_energy: torch.Tensor,
    js_raw: torch.Tensor,
    kl_raw: torch.Tensor,
    rkl_raw: torch.Tensor,
    kl_asym_raw: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """
    Oracle branch labels for supervised routing.

    Branch order:
        0 boost
        1 damp
        2 fallback
        3 neutral

    Logic:
        - fallback: spectral mismatch / asymmetric distribution damage
        - boost: direction is truthful and alpha_energy says correction can reduce energy
        - damp: wrong direction / overstep / waste
        - neutral: weak or ambiguous sample
    """
    eps = cfg.eps

    # If correction points opposite to residual, y = y0 + alpha*delta reduces residual.
    # Therefore inner < 0 is a truthful correction direction.
    inner = (r_base * delta_raw).mean(dim=(1, 2), keepdim=True)
    d_energy = delta_raw.pow(2).mean(dim=(1, 2), keepdim=True) + eps
    r_energy = r_base.pow(2).mean(dim=(1, 2), keepdim=True) + eps

    direction_good = inner < 0
    direction_bad = inner > 0

    corr_strength = d_energy / r_energy
    understep = (alpha_energy > 0.25 * cfg.alpha_max) & direction_good
    weak = corr_strength < 1e-4
    overstep = corr_strength > 2.5

    # Strong spectral mismatch: if raw correction redistributes residual spectrum too much,
    # do not allow simple boost.
    spectral_bad = (js_raw > cfg.js_limit) | ((kl_raw > rkl_raw) & (kl_asym_raw > 0.05) & (js_raw > 0.05))

    labels = torch.full((r_base.size(0),), 3, dtype=torch.long, device=r_base.device)

    labels[spectral_bad.view(-1)] = 2
    labels[(direction_bad | overstep).view(-1)] = 1
    labels[(understep & (~spectral_bad) & (~weak)).view(-1)] = 0

    return labels



def boltzmann_tail_mass(
    residual: torch.Tensor,
    tail_ratio: float = 1.0 / 3.0,
    temperature: float = 0.20,
    eps: float = 1e-6,
) -> torch.Tensor:
    spec = torch.fft.rfft(residual, dim=1, norm="ortho")
    energy = (spec.real.pow(2) + spec.imag.pow(2)).mean(dim=-1)

    e_min = energy.min(dim=-1, keepdim=True).values
    e_max = energy.max(dim=-1, keepdim=True).values
    hard = (energy - e_min) / (e_max - e_min + eps)

    b = torch.softmax(hard / max(temperature, eps), dim=-1)
    f = b.size(-1)
    start = int((1.0 - tail_ratio) * f)
    start = min(max(start, 0), f - 1)
    return b[:, start:].sum(dim=-1).mean()


class ResidualStateEstimator(nn.Module):
    """
    Estimate residual state from input and initial forecast.
    Branch order:
        0 boost
        1 damp
        2 fallback
        3 neutral
    """

    def __init__(self, cfg: TGCAConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.enc_in
        in_dim = 14 * c

        self.state_mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.d_state),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_state, cfg.d_state),
            nn.GELU(),
        )

        self.proxy_conv = nn.Sequential(
            nn.Conv1d(c, c, kernel_size=3, padding=1, groups=c),
            nn.GELU(),
            nn.Conv1d(c, c, kernel_size=1),
        )

        self.proxy_gain = nn.Linear(cfg.d_state, c)
        self.branch_head = nn.Linear(cfg.d_state, cfg.n_branches)
        self.op_head = nn.Linear(cfg.d_state, cfg.n_ops)
        self.alpha_head = nn.Linear(cfg.d_state, 1)

    @staticmethod
    def _stats(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        xmin = x.min(dim=1).values
        xmax = x.max(dim=1).values
        first = x[:, 0, :]
        last = x[:, -1, :]
        slope = last - first
        return torch.cat([mean, std, xmin, xmax, first, last, slope], dim=-1)

    def forward(self, x: torch.Tensor, y0: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = torch.cat([self._stats(x), self._stats(y0)], dim=-1)
        z = self.state_mlp(feat)

        proxy = self.proxy_conv(y0.transpose(1, 2)).transpose(1, 2)
        gain = torch.tanh(self.proxy_gain(z)).unsqueeze(1)
        r_proxy = 0.10 * torch.tanh(proxy) + 0.05 * gain * torch.tanh(y0)

        return {
            "z": z,
            "r_proxy": r_proxy,
            "branch_logits": self.branch_head(z),
            "op_logits": self.op_head(z),
            "alpha_raw": torch.sigmoid(self.alpha_head(z)).view(-1, 1, 1),
        }


class CorrectionOperatorBank(nn.Module):
    """
    TGCA correction bank:
        0 amplitude
        1 bias
        2 smooth
        3 frequency shrink / low-pass
        4 segment
        5 local convolution
    """

    def __init__(self, cfg: TGCAConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.enc_in
        h = cfg.pred_len
        f = h // 2 + 1

        self.smooth = MovingAverage(kernel_size=7)
        self.amp_head = nn.Linear(cfg.d_state, c)
        self.bias_head = nn.Linear(cfg.d_state, c)
        self.segment_head = nn.Linear(cfg.d_state, cfg.n_segments * c)

        base = torch.linspace(2.0, -2.0, f).view(f, 1)
        self.freq_gate_logits = nn.Parameter(base)

        self.local_conv = nn.Conv1d(c, c, kernel_size=3, padding=1, groups=c)
        self.out_norm = nn.LayerNorm(c)

    def _segment_delta(self, y0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        b, h, c = y0.shape
        seg = torch.tanh(self.segment_head(z)).view(b, self.cfg.n_segments, c)
        gate = torch.zeros_like(y0)
        for i in range(self.cfg.n_segments):
            s = int(i * h / self.cfg.n_segments)
            e = int((i + 1) * h / self.cfg.n_segments)
            gate[:, s:e, :] = seg[:, i, :].unsqueeze(1)
        return gate * y0

    def _frequency_delta(self, y0: torch.Tensor) -> torch.Tensor:
        b, h, c = y0.shape
        spec = torch.fft.rfft(y0, dim=1, norm="ortho")
        f = spec.size(1)
        mask = torch.sigmoid(self.freq_gate_logits[:f]).view(1, f, 1)
        filtered = torch.fft.irfft(spec * mask, n=h, dim=1, norm="ortho")
        return filtered - y0

    def forward(self, y0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        amp = torch.tanh(self.amp_head(z)).unsqueeze(1) * y0

        bias = torch.tanh(self.bias_head(z)).unsqueeze(1)
        bias = bias.expand_as(y0)

        smooth = self.smooth(y0) - y0
        freq = self._frequency_delta(y0)
        segment = self._segment_delta(y0, z)
        local = self.local_conv(y0.transpose(1, 2)).transpose(1, 2) - y0

        deltas = torch.stack([amp, bias, smooth, freq, segment, local], dim=1)
        return self.cfg.delta_scale * torch.tanh(self.out_norm(deltas))


class TGCARouter(nn.Module):
    def __init__(self, cfg: TGCAConfig):
        super().__init__()
        self.cfg = cfg
        caps = torch.tensor(
            [cfg.alpha_max, cfg.alpha_damp_cap, cfg.alpha_fallback, cfg.alpha_neutral_cap],
            dtype=torch.float32,
        )
        self.register_buffer("branch_caps", caps)

    def forward(self, branch_logits: torch.Tensor, op_logits: torch.Tensor):
        branch_probs = torch.softmax(branch_logits / self.cfg.router_tau, dim=-1)
        op_probs = torch.softmax(op_logits / self.cfg.router_tau, dim=-1)
        alpha_branch = (branch_probs * self.branch_caps.view(1, -1)).sum(dim=-1).view(-1, 1, 1)
        return branch_probs, op_probs, alpha_branch


class HardTrustProjector(nn.Module):
    def __init__(self, cfg: TGCAConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        y0: torch.Tensor,
        delta_raw: torch.Tensor,
        r_proxy: torch.Tensor,
        alpha_raw: torch.Tensor,
        alpha_branch: torch.Tensor,
        y_true_norm: Optional[torch.Tensor] = None,
    ):
        eps = self.cfg.eps
        r_ref = y0 - y_true_norm if y_true_norm is not None else r_proxy

        inner = (r_ref * delta_raw).mean(dim=(1, 2), keepdim=True)
        delta_energy = delta_raw.pow(2).mean(dim=(1, 2), keepdim=True) + eps
        alpha_energy = (-2.0 * inner / delta_energy).clamp(0.0, self.cfg.alpha_max)

        # LOWMEM_FAST_PATH_NO_TRUE:
        # In the standard TSLib/DistDF training loop, y_true is not passed into model.forward.
        # Do NOT use alpha_energy here. At initialization r_proxy can be wrong, causing
        # alpha_energy to clamp to 0 and making TGCA identical to the base forecaster.
        # Here we allow a small trainable correction path; full energy/spectral trust
        # will be enabled later with y_true + custom TGCA loss.
        if y_true_norm is None:
            z = torch.zeros_like(alpha_raw)
            alpha_spec = torch.full_like(alpha_raw, self.cfg.alpha_max)

            alpha_raw = self.cfg.alpha_max * alpha_raw
            alpha_train_cap = torch.full_like(alpha_raw, min(0.15, self.cfg.alpha_max))

            alpha = torch.minimum(alpha_raw, alpha_branch)
            alpha = torch.minimum(alpha, alpha_spec)
            alpha = torch.minimum(alpha, alpha_train_cap)

            y = y0 + alpha * delta_raw

            return y, {
                "alpha": alpha,
                "alpha_raw": alpha_raw,
                "alpha_energy": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "alpha_branch": alpha_branch,
                "alpha_spec": alpha_spec,
                "js_raw": z,
                "kl_raw": z,
                "rkl_raw": z,
                "kl_asym_raw": z,
                "spectral_bad": z,
            }

        p0 = spectral_prob(r_ref, eps=eps)
        p1 = spectral_prob(r_ref + delta_raw, eps=eps)
        js = js_div(p0, p1, eps=eps).view(-1, 1, 1)
        kl01 = kl_div(p0, p1, eps=eps).view(-1, 1, 1)
        kl10 = kl_div(p1, p0, eps=eps).view(-1, 1, 1)
        kl_asym = kl01 - kl10

        spectral_bad = js > self.cfg.js_limit
        alpha_spec = torch.where(
            spectral_bad,
            torch.full_like(alpha_raw, self.cfg.alpha_fallback),
            torch.full_like(alpha_raw, self.cfg.alpha_max),
        )

        alpha_raw = self.cfg.alpha_max * alpha_raw
        alpha = torch.minimum(alpha_raw, alpha_energy)
        alpha = torch.minimum(alpha, alpha_branch)
        alpha = torch.minimum(alpha, alpha_spec)

        y = y0 + alpha * delta_raw

        return y, {
            "alpha": alpha,
            "alpha_raw": alpha_raw,
            "alpha_energy": alpha_energy,
            "alpha_branch": alpha_branch,
            "alpha_spec": alpha_spec,
            "js_raw": js,
            "kl_raw": kl01,
            "rkl_raw": kl10,
            "kl_asym_raw": kl_asym,
            "spectral_bad": spectral_bad.float(),
        }


class TGCAForecaster(nn.Module):
    def __init__(self, cfg: TGCAConfig):
        super().__init__()
        self.cfg = cfg
        self.base = DecompositionLinearBase(cfg)
        self.state = ResidualStateEstimator(cfg)
        self.bank = CorrectionOperatorBank(cfg)
        self.router = TGCARouter(cfg)
        self.projector = HardTrustProjector(cfg)

    def _norm(self, x: torch.Tensor):
        if not self.cfg.use_revin:
            mean = torch.zeros_like(x[:, :1, :])
            std = torch.ones_like(x[:, :1, :])
            return x, mean, std
        center_mode = os.environ.get('TGCA_EXCHANGE_BASE_CENTER', 'mean')
        if center_mode == 'last':
            mean = x[:, -1:, :].detach()
        elif center_mode == 'mean':
            mean = x.mean(dim=1, keepdim=True).detach()
        else:
            raise ValueError(f'unknown base center mode: {center_mode}')
        var = x.var(dim=1, keepdim=True, unbiased=False).detach()
        std = torch.sqrt(var + self.cfg.eps)
        return (x - mean) / std, mean, std

    @staticmethod
    def _denorm(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        return y * std + mean

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        y_true: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ):
        x_norm, mean, std = self._norm(x_enc)
        y0_norm = self.base(x_norm)

        y_true_norm = None
        if y_true is not None:
            y_true_norm = (y_true - mean) / std

        s = self.state(x_norm, y0_norm)
        branch_probs, op_probs, alpha_branch = self.router(s["branch_logits"], s["op_logits"])

        deltas = self.bank(y0_norm, s["z"])
        delta_raw = (deltas * op_probs[:, :, None, None]).sum(dim=1)

        y_norm, trust = self.projector(
            y0=y0_norm,
            delta_raw=delta_raw,
            r_proxy=s["r_proxy"],
            alpha_raw=s["alpha_raw"],
            alpha_branch=alpha_branch,
            y_true_norm=y_true_norm,
        )

        y = self._denorm(y_norm, mean, std)

        if not return_aux:
            return y

        y0 = self._denorm(y0_norm, mean, std)

        aux = {
            "y0": y0,
            "y_norm": y_norm,
            "y0_norm": y0_norm,
            "delta_raw_norm": delta_raw,
            "r_proxy": s["r_proxy"] * std,
            "r_proxy_norm": s["r_proxy"],
            "branch_probs": branch_probs,
            "op_probs": op_probs,
            "deltas_norm": deltas,
            "mean": mean,
            "std": std,
        }
        aux.update(trust)
        return y, aux



def tgca_forecasting_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    beta_mae: float = 1.0,
    lambda_energy: float = 0.20,
    lambda_state: float = 0.05,
    lambda_spec: float = 0.05,
    lambda_tail: float = 0.05,
    lambda_alpha: float = 0.001,
    lambda_branch: float = 0.05,
    lambda_curv: float = 0.02,
    lambda_segment: float = 0.05,
    margin: float = 0.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    TGCA-Forecaster v2 loss.

    Includes:
      - forecast MSE + MAE
      - residual energy hinge
      - residual-state estimator loss
      - spectral JS movement loss
      - Boltzmann tail hard-mode loss
      - alpha regularization
      - branch oracle supervision, if branch_oracle exists
      - curvature proxy penalty
      - segment-wise energy safety loss
    """

    y0 = aux["y0"]

    r_base = y0 - y_true
    r_final = y_pred - y_true

    mse = F.mse_loss(y_pred, y_true)
    mae = F.l1_loss(y_pred, y_true)
    forecast_loss = mse + beta_mae * mae

    base_energy = r_base.pow(2).mean()
    final_energy = r_final.pow(2).mean()
    energy_hinge = F.relu(
        torch.log(final_energy + eps) - torch.log(base_energy + eps) + margin
    )

    r_proxy = aux.get("r_proxy", torch.zeros_like(r_base))
    state_loss = F.smooth_l1_loss(r_proxy, r_base.detach())

    p_base = spectral_prob(r_base.detach(), eps=eps)
    p_final = spectral_prob(r_final, eps=eps)
    spec_js = js_div(p_base, p_final, eps=eps).mean()

    tail_base = boltzmann_tail_mass(r_base.detach(), eps=eps)
    tail_final = boltzmann_tail_mass(r_final, eps=eps)
    tail_loss = F.relu(tail_final - tail_base)

    alpha = aux.get("alpha", torch.zeros(y_pred.size(0), 1, 1, device=y_pred.device))
    alpha_loss = alpha.pow(2).mean()

    # Branch oracle supervision. Always define these variables.
    branch_loss = torch.zeros((), device=y_pred.device)
    oracle_boost_rate = torch.zeros((), device=y_pred.device)
    oracle_damp_rate = torch.zeros((), device=y_pred.device)
    oracle_fallback_rate = torch.zeros((), device=y_pred.device)
    oracle_neutral_rate = torch.zeros((), device=y_pred.device)

    if "branch_oracle" in aux and "branch_probs" in aux:
        branch_probs = aux["branch_probs"].clamp_min(eps)
        branch_oracle = aux["branch_oracle"].long()
        branch_loss = F.nll_loss(branch_probs.log(), branch_oracle)

        oracle_boost_rate = (branch_oracle == 0).float().mean()
        oracle_damp_rate = (branch_oracle == 1).float().mean()
        oracle_fallback_rate = (branch_oracle == 2).float().mean()
        oracle_neutral_rate = (branch_oracle == 3).float().mean()

    # Curvature / condition proxy.
    curvature_proxy = aux.get("curvature_proxy", torch.zeros_like(alpha))
    curv_loss = F.relu(curvature_proxy - 2.5).mean()

    # Segment-wise safety loss: front/middle/back should not jointly worsen.
    try:
        seg_ratio = segment_energy_ratio(r_base.detach(), r_final, n_segments=3, eps=eps)
        segment_loss = F.relu(seg_ratio).mean()
    except Exception:
        segment_loss = torch.zeros((), device=y_pred.device)

    loss = (
        forecast_loss
        + lambda_energy * energy_hinge
        + lambda_state * state_loss
        + lambda_spec * spec_js
        + lambda_tail * tail_loss
        + lambda_alpha * alpha_loss
        + lambda_branch * branch_loss
        + lambda_curv * curv_loss
        + lambda_segment * segment_loss
    )

    with torch.no_grad():
        branch_probs = aux.get("branch_probs", torch.zeros(y_pred.size(0), 4, device=y_pred.device))
        spectral_bad = aux.get("spectral_bad", torch.zeros_like(alpha))
        direction_cos = aux.get("direction_cos", torch.zeros_like(alpha))

        logs = {
            "loss": float(loss.detach().cpu()),
            "mse": float(mse.detach().cpu()),
            "mae": float(mae.detach().cpu()),
            "base_mse": float(F.mse_loss(y0, y_true).detach().cpu()),
            "base_mae": float(F.l1_loss(y0, y_true).detach().cpu()),
            "energy_hinge": float(energy_hinge.detach().cpu()),
            "state_loss": float(state_loss.detach().cpu()),
            "spec_js": float(spec_js.detach().cpu()),
            "tail_base": float(tail_base.detach().cpu()),
            "tail_final": float(tail_final.detach().cpu()),
            "alpha_mean": float(alpha.mean().detach().cpu()),
            "alpha_max": float(alpha.max().detach().cpu()),
            "boost_rate": float(branch_probs[:, 0].mean().detach().cpu()) if branch_probs.ndim == 2 and branch_probs.size(1) > 0 else 0.0,
            "damp_rate": float(branch_probs[:, 1].mean().detach().cpu()) if branch_probs.ndim == 2 and branch_probs.size(1) > 1 else 0.0,
            "fallback_rate": float(branch_probs[:, 2].mean().detach().cpu()) if branch_probs.ndim == 2 and branch_probs.size(1) > 2 else 0.0,
            "neutral_rate": float(branch_probs[:, 3].mean().detach().cpu()) if branch_probs.ndim == 2 and branch_probs.size(1) > 3 else 0.0,
            "spectral_bad_rate": float(spectral_bad.mean().detach().cpu()),
            "branch_loss": float(branch_loss.detach().cpu()),
            "curv_loss": float(curv_loss.detach().cpu()),
            "segment_loss": float(segment_loss.detach().cpu()),
            "curvature_proxy": float(curvature_proxy.mean().detach().cpu()),
            "direction_cos": float(direction_cos.mean().detach().cpu()),
            "oracle_boost_rate": float(oracle_boost_rate.detach().cpu()),
            "oracle_damp_rate": float(oracle_damp_rate.detach().cpu()),
            "oracle_fallback_rate": float(oracle_fallback_rate.detach().cpu()),
            "oracle_neutral_rate": float(oracle_neutral_rate.detach().cpu()),
        }

    return loss, logs


class Model(nn.Module):
    """
    Wrapper for Time-Series-Library / DistDF-style run.py.

    Default forward returns y_pred only.
    For custom TGCA loss, use:
        pred, aux = model.core(batch_x, y_true=batch_y[:, -pred_len:, :], return_aux=True)
        loss, logs = tgca_forecasting_loss(pred, target, aux)
    """

    def __init__(self, configs):
        super().__init__()

        enc_in = getattr(configs, "enc_in", None)
        if enc_in is None:
            enc_in = getattr(configs, "c_out", None)
        if enc_in is None:
            raise ValueError("configs must contain enc_in or c_out")

        cfg = TGCAConfig(
            seq_len=getattr(configs, "seq_len"),
            pred_len=getattr(configs, "pred_len"),
            enc_in=enc_in,
            moving_avg=getattr(configs, "moving_avg", 25),
            individual=getattr(configs, "individual", False),
            d_state=getattr(configs, "d_model", 128),
            dropout=getattr(configs, "dropout", 0.10),
            alpha_max=getattr(configs, "tgca_alpha_max", 0.80),
            alpha_fallback=getattr(configs, "tgca_alpha_fallback", 0.05),
            js_limit=getattr(configs, "tgca_js_limit", 0.20),
            delta_scale=getattr(configs, "tgca_delta_scale", 0.20),
            use_revin=getattr(configs, "use_revin", True),
        )
        self.core = TGCAForecaster(cfg)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.core(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            mask=mask,
            y_true=None,
            return_aux=False,
        )




# ===== TGCA_V2_FORCED_PATCH_START =====
# This block force-overrides HardTrustProjector and TGCAForecaster.forward
# without relying on fragile source-block replacement anchors.

def _tgca_safe_cosine_v2(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = (a * b).sum(dim=(1, 2), keepdim=True)
    den = torch.sqrt(a.pow(2).sum(dim=(1, 2), keepdim=True) + eps) * torch.sqrt(b.pow(2).sum(dim=(1, 2), keepdim=True) + eps)
    return num / den.clamp_min(eps)


def _tgca_segment_energy_ratio_v2(
    r_base: torch.Tensor,
    r_final: torch.Tensor,
    n_segments: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    b, h, c = r_base.shape
    outs = []
    for i in range(n_segments):
        st = int(i * h / n_segments)
        ed = int((i + 1) * h / n_segments)
        eb = r_base[:, st:ed, :].pow(2).mean(dim=(1, 2))
        ef = r_final[:, st:ed, :].pow(2).mean(dim=(1, 2))
        outs.append(torch.log((ef + eps) / (eb + eps)))
    return torch.stack(outs, dim=-1)


def _tgca_make_branch_oracle_v2(
    r_base: torch.Tensor,
    delta_raw: torch.Tensor,
    alpha_energy: torch.Tensor,
    js_raw: torch.Tensor,
    kl_raw: torch.Tensor,
    rkl_raw: torch.Tensor,
    kl_asym_raw: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """
    Branch order:
      0 boost
      1 damp
      2 fallback
      3 neutral
    """
    eps = cfg.eps

    inner = (r_base * delta_raw).mean(dim=(1, 2), keepdim=True)
    d_energy = delta_raw.pow(2).mean(dim=(1, 2), keepdim=True) + eps
    r_energy = r_base.pow(2).mean(dim=(1, 2), keepdim=True) + eps

    direction_good = inner < 0
    direction_bad = inner > 0

    corr_strength = d_energy / r_energy
    weak = corr_strength < 1e-5
    overstep = corr_strength > 2.5

    understep = (alpha_energy > 0.20 * cfg.alpha_max) & direction_good

    spectral_bad = (
        (js_raw > cfg.js_limit)
        | ((kl_raw > rkl_raw) & (kl_asym_raw > 0.05) & (js_raw > 0.05))
    )

    labels = torch.full((r_base.size(0),), 3, dtype=torch.long, device=r_base.device)

    labels[spectral_bad.view(-1)] = 2
    labels[(direction_bad | overstep).view(-1)] = 1
    labels[(understep & (~spectral_bad) & (~weak)).view(-1)] = 0

    return labels


# 让已有 tgca_forecasting_loss 里调用 segment_energy_ratio 时能找到 V2 版本
segment_energy_ratio = _tgca_segment_energy_ratio_v2
safe_cosine = _tgca_safe_cosine_v2
make_tgca_branch_oracle = _tgca_make_branch_oracle_v2


class HardTrustProjectorV2(nn.Module):
    """
    V2 hard-projected trust-region projector.

    Training path with y_true:
      - energy-safe alpha
      - spectral hard gate
      - curvature proxy cap
      - direction truthfulness cap

    No-y_true path:
      - low-memory small trainable correction path
    """

    def __init__(self, cfg: TGCAConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        y0: torch.Tensor,
        delta_raw: torch.Tensor,
        r_proxy: torch.Tensor,
        alpha_raw: torch.Tensor,
        alpha_branch: torch.Tensor,
        y_true_norm: Optional[torch.Tensor] = None,
    ):
        eps = self.cfg.eps

        r_ref = y0 - y_true_norm if y_true_norm is not None else r_proxy

        inner = (r_ref * delta_raw).mean(dim=(1, 2), keepdim=True)
        delta_energy = delta_raw.pow(2).mean(dim=(1, 2), keepdim=True) + eps
        residual_energy = r_ref.pow(2).mean(dim=(1, 2), keepdim=True) + eps

        alpha_energy = (-2.0 * inner / delta_energy).clamp(0.0, self.cfg.alpha_max)

        # 普通 run.py 没有 y_true，不要用 alpha_energy 直接卡死修正。
        if y_true_norm is None:
            z = torch.zeros_like(alpha_raw)
            alpha_raw_scaled = self.cfg.alpha_max * alpha_raw
            alpha_train_cap = torch.full_like(alpha_raw, min(0.15, self.cfg.alpha_max))

            alpha = torch.minimum(alpha_raw_scaled, alpha_branch)
            alpha = torch.minimum(alpha, alpha_train_cap)

            y = y0 + alpha * delta_raw

            return y, {
                "alpha": alpha,
                "alpha_raw": alpha_raw_scaled,
                "alpha_energy": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "alpha_branch": alpha_branch,
                "alpha_spec": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "alpha_curv": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "alpha_dir": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "js_raw": z,
                "kl_raw": z,
                "rkl_raw": z,
                "kl_asym_raw": z,
                "spectral_bad": z,
                "curvature_proxy": z,
                "direction_cos": z,
            }

        # full TGCA path
        p0 = spectral_prob(r_ref, eps=eps)
        p1 = spectral_prob(r_ref + delta_raw, eps=eps)

        js = js_div(p0, p1, eps=eps).view(-1, 1, 1)
        kl01 = kl_div(p0, p1, eps=eps).view(-1, 1, 1)
        kl10 = kl_div(p1, p0, eps=eps).view(-1, 1, 1)
        kl_asym = kl01 - kl10

        spectral_bad = (
            (js > self.cfg.js_limit)
            | ((kl01 > kl10) & (kl_asym > 0.05) & (js > 0.05))
        )

        alpha_spec = torch.where(
            spectral_bad,
            torch.full_like(alpha_raw, self.cfg.alpha_fallback),
            torch.full_like(alpha_raw, self.cfg.alpha_max),
        )

        curvature_proxy = delta_energy / residual_energy
        alpha_curv = torch.where(
            curvature_proxy > 2.5,
            torch.full_like(alpha_raw, min(self.cfg.alpha_damp_cap, self.cfg.alpha_max)),
            torch.full_like(alpha_raw, self.cfg.alpha_max),
        )

        direction_cos = _tgca_safe_cosine_v2(r_ref, delta_raw, eps=eps)
        direction_bad = direction_cos > 0.0

        alpha_dir = torch.where(
            direction_bad,
            torch.full_like(alpha_raw, self.cfg.alpha_damp_cap),
            torch.full_like(alpha_raw, self.cfg.alpha_max),
        )

        alpha_raw_scaled = self.cfg.alpha_max * alpha_raw

        alpha = torch.minimum(alpha_raw_scaled, alpha_energy)
        alpha = torch.minimum(alpha, alpha_branch)
        alpha = torch.minimum(alpha, alpha_spec)
        alpha = torch.minimum(alpha, alpha_curv)
        alpha = torch.minimum(alpha, alpha_dir)

        y = y0 + alpha * delta_raw

        return y, {
            "alpha": alpha,
            "alpha_raw": alpha_raw_scaled,
            "alpha_energy": alpha_energy,
            "alpha_branch": alpha_branch,
            "alpha_spec": alpha_spec,
            "alpha_curv": alpha_curv,
            "alpha_dir": alpha_dir,
            "js_raw": js,
            "kl_raw": kl01,
            "rkl_raw": kl10,
            "kl_asym_raw": kl_asym,
            "spectral_bad": spectral_bad.float(),
            "curvature_proxy": curvature_proxy,
            "direction_cos": direction_cos,
        }


def _tgca_forward_v2(
    self,
    x_enc: torch.Tensor,
    x_mark_enc: Optional[torch.Tensor] = None,
    x_dec: Optional[torch.Tensor] = None,
    x_mark_dec: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    y_true: Optional[torch.Tensor] = None,
    return_aux: bool = True,
):
    x_norm, mean, std = self._norm(x_enc)

    y0_norm = self.base(x_norm)

    if y_true is not None:
        y_true_norm = (y_true - mean) / std
    else:
        y_true_norm = None

    state = self.state(x_norm, y0_norm)

    branch_probs, op_probs, alpha_branch = self.router(
        state["branch_logits"],
        state["op_logits"],
    )

    deltas = self.bank(y0_norm, state["z"])
    delta_raw = (deltas * op_probs[:, :, None, None]).sum(dim=1)

    y_norm, trust_info = self.projector(
        y0=y0_norm,
        delta_raw=delta_raw,
        r_proxy=state["r_proxy"],
        alpha_raw=state["alpha_raw"],
        alpha_branch=alpha_branch,
        y_true_norm=y_true_norm,
    )

    if y_true_norm is not None:
        r_base_norm = y0_norm - y_true_norm
        trust_info["branch_oracle"] = _tgca_make_branch_oracle_v2(
            r_base=r_base_norm,
            delta_raw=delta_raw,
            alpha_energy=trust_info["alpha_energy"],
            js_raw=trust_info["js_raw"],
            kl_raw=trust_info["kl_raw"],
            rkl_raw=trust_info["rkl_raw"],
            kl_asym_raw=trust_info["kl_asym_raw"],
            cfg=self.cfg,
        )

    y = self._denorm(y_norm, mean, std)
    y0 = self._denorm(y0_norm, mean, std)
    r_proxy = state["r_proxy"] * std

    if not return_aux:
        return y

    aux = {
        "y0": y0,
        "y_norm": y_norm,
        "y0_norm": y0_norm,
        "delta_raw_norm": delta_raw,
        "r_proxy": r_proxy,
        "r_proxy_norm": state["r_proxy"],
        "branch_probs": branch_probs,
        "op_probs": op_probs,
        "deltas_norm": deltas,
        "mean": mean,
        "std": std,
    }
    aux.update(trust_info)

    return y, aux


# Rebind global name used by TGCAForecaster.__init__ at runtime.
HardTrustProjector = HardTrustProjectorV2
TGCAForecaster.forward = _tgca_forward_v2

# ===== TGCA_V2_FORCED_PATCH_END =====




# ===== TGCA_V3_SOFT_REAL_START =====
# V3 goal:
#   Keep v1's strong correction performance,
#   keep v2's mechanism signals,
#   but convert hard branch/direction constraints into soft mechanism regularization.

def _tgca_soft_branch_target_v3(hard_oracle: torch.Tensor, device=None) -> torch.Tensor:
    """
    Convert hard branch labels into soft branch targets.

    Branch order:
      0 boost
      1 damp
      2 fallback
      3 neutral

    Design:
      - boost label still prefers boost, but leaves mass for neutral/damp
      - damp label no longer kills boost completely
      - fallback label is the only branch with strong fallback preference
      - neutral label allows safe boost if forecasting loss supports it
    """
    if device is None:
        device = hard_oracle.device

    table = torch.tensor([
        [0.65, 0.15, 0.05, 0.15],  # boost
        [0.35, 0.35, 0.05, 0.25],  # damp, soft not hard
        [0.10, 0.15, 0.60, 0.15],  # fallback
        [0.35, 0.15, 0.05, 0.45],  # neutral
    ], dtype=torch.float32, device=device)

    return table[hard_oracle.long()]


class HardTrustProjectorV3(nn.Module):
    """
    V3 soft-real trust projector.

    Differences from V2:
      1. alpha_branch is given a floor, so branch router cannot over-suppress useful correction.
      2. direction cap only triggers for severe wrong-direction cases.
      3. curvature cap only triggers for large correction/residual energy ratio.
      4. spectral hard gate remains hard, because true spectral mismatch should still fallback.
    """

    def __init__(self, cfg: TGCAConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        y0: torch.Tensor,
        delta_raw: torch.Tensor,
        r_proxy: torch.Tensor,
        alpha_raw: torch.Tensor,
        alpha_branch: torch.Tensor,
        y_true_norm: Optional[torch.Tensor] = None,
    ):
        eps = self.cfg.eps
        r_ref = y0 - y_true_norm if y_true_norm is not None else r_proxy

        inner = (r_ref * delta_raw).mean(dim=(1, 2), keepdim=True)
        delta_energy = delta_raw.pow(2).mean(dim=(1, 2), keepdim=True) + eps
        residual_energy = r_ref.pow(2).mean(dim=(1, 2), keepdim=True) + eps

        alpha_energy = (-2.0 * inner / delta_energy).clamp(0.0, self.cfg.alpha_max)

        # run.py no-y_true path: keep small trainable correction.
        if y_true_norm is None:
            z = torch.zeros_like(alpha_raw)
            alpha_raw_scaled = self.cfg.alpha_max * alpha_raw
            alpha_train_cap = torch.full_like(alpha_raw, min(0.15, self.cfg.alpha_max))
            alpha = torch.minimum(alpha_raw_scaled, alpha_branch)
            alpha = torch.minimum(alpha, alpha_train_cap)
            y = y0 + alpha * delta_raw

            return y, {
                "alpha": alpha,
                "alpha_raw": alpha_raw_scaled,
                "alpha_energy": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "alpha_branch": alpha_branch,
                "alpha_branch_eff": alpha_branch,
                "alpha_spec": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "alpha_curv": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "alpha_dir": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "js_raw": z,
                "kl_raw": z,
                "rkl_raw": z,
                "kl_asym_raw": z,
                "spectral_bad": z,
                "curvature_proxy": z,
                "direction_cos": z,
                "severe_direction_bad": z,
            }

        # full TGCA path
        p0 = spectral_prob(r_ref, eps=eps)
        p1 = spectral_prob(r_ref + delta_raw, eps=eps)

        js = js_div(p0, p1, eps=eps).view(-1, 1, 1)
        kl01 = kl_div(p0, p1, eps=eps).view(-1, 1, 1)
        kl10 = kl_div(p1, p0, eps=eps).view(-1, 1, 1)
        kl_asym = kl01 - kl10

        spectral_bad = (
            (js > self.cfg.js_limit)
            | ((kl01 > kl10) & (kl_asym > 0.05) & (js > 0.05))
        )

        alpha_spec = torch.where(
            spectral_bad,
            torch.full_like(alpha_raw, self.cfg.alpha_fallback),
            torch.full_like(alpha_raw, self.cfg.alpha_max),
        )

        curvature_proxy = delta_energy / residual_energy

        # V3: only very large correction energy triggers curvature damping.
        alpha_curv = torch.where(
            curvature_proxy > 5.0,
            torch.full_like(alpha_raw, min(self.cfg.alpha_damp_cap, self.cfg.alpha_max)),
            torch.full_like(alpha_raw, self.cfg.alpha_max),
        )

        direction_cos = _tgca_safe_cosine_v2(r_ref, delta_raw, eps=eps) if "_tgca_safe_cosine_v2" in globals() else safe_cosine(r_ref, delta_raw, eps=eps)

        # V3: direction damping only for severe wrong direction.
        severe_direction_bad = (direction_cos > 0.35) & (alpha_energy < 0.05 * self.cfg.alpha_max)

        alpha_dir = torch.where(
            severe_direction_bad,
            torch.full_like(alpha_raw, min(0.35 * self.cfg.alpha_max, self.cfg.alpha_max)),
            torch.full_like(alpha_raw, self.cfg.alpha_max),
        )

        alpha_raw_scaled = self.cfg.alpha_max * alpha_raw

        # V3 key: branch cap becomes a soft floor, not a hard suppressor.
        branch_floor = torch.full_like(alpha_branch, 0.60 * self.cfg.alpha_max)
        alpha_branch_eff = torch.maximum(alpha_branch, branch_floor)

        # But if spectral_bad, fallback still wins through alpha_spec.
        alpha = torch.minimum(alpha_raw_scaled, alpha_energy)
        alpha = torch.minimum(alpha, alpha_branch_eff)
        alpha = torch.minimum(alpha, alpha_spec)
        alpha = torch.minimum(alpha, alpha_curv)
        alpha = torch.minimum(alpha, alpha_dir)

        y = y0 + alpha * delta_raw

        return y, {
            "alpha": alpha,
            "alpha_raw": alpha_raw_scaled,
            "alpha_energy": alpha_energy,
            "alpha_branch": alpha_branch,
            "alpha_branch_eff": alpha_branch_eff,
            "alpha_spec": alpha_spec,
            "alpha_curv": alpha_curv,
            "alpha_dir": alpha_dir,
            "js_raw": js,
            "kl_raw": kl01,
            "rkl_raw": kl10,
            "kl_asym_raw": kl_asym,
            "spectral_bad": spectral_bad.float(),
            "curvature_proxy": curvature_proxy,
            "direction_cos": direction_cos,
            "severe_direction_bad": severe_direction_bad.float(),
        }


def tgca_forecasting_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    beta_mae: float = 1.0,
    lambda_energy: float = 0.20,
    lambda_state: float = 0.05,
    lambda_spec: float = 0.05,
    lambda_tail: float = 0.05,
    lambda_alpha: float = 0.001,
    lambda_branch: float = 0.02,
    lambda_curv: float = 0.02,
    lambda_segment: float = 0.05,
    margin: float = 0.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    V3 soft-real loss:
      - same forecasting / energy / state / spectral / tail losses
      - soft branch KL instead of hard branch NLL
      - curvature and segment safety remain auxiliary
    """

    y0 = aux["y0"]
    r_base = y0 - y_true
    r_final = y_pred - y_true

    mse = F.mse_loss(y_pred, y_true)
    mae = F.l1_loss(y_pred, y_true)
    forecast_loss = mse + beta_mae * mae

    base_energy = r_base.pow(2).mean()
    final_energy = r_final.pow(2).mean()
    energy_hinge = F.relu(torch.log(final_energy + eps) - torch.log(base_energy + eps) + margin)

    r_proxy = aux.get("r_proxy", torch.zeros_like(r_base))
    state_loss = F.smooth_l1_loss(r_proxy, r_base.detach())

    p_base = spectral_prob(r_base.detach(), eps=eps)
    p_final = spectral_prob(r_final, eps=eps)
    spec_js = js_div(p_base, p_final, eps=eps).mean()

    tail_base = boltzmann_tail_mass(r_base.detach(), eps=eps)
    tail_final = boltzmann_tail_mass(r_final, eps=eps)
    tail_loss = F.relu(tail_final - tail_base)

    alpha = aux.get("alpha", torch.zeros(y_pred.size(0), 1, 1, device=y_pred.device))
    alpha_loss = alpha.pow(2).mean()

    branch_loss = torch.zeros((), device=y_pred.device)
    oracle_boost_rate = torch.zeros((), device=y_pred.device)
    oracle_damp_rate = torch.zeros((), device=y_pred.device)
    oracle_fallback_rate = torch.zeros((), device=y_pred.device)
    oracle_neutral_rate = torch.zeros((), device=y_pred.device)

    if "branch_oracle" in aux and "branch_probs" in aux:
        branch_oracle = aux["branch_oracle"].long()
        branch_probs = aux["branch_probs"].clamp_min(eps)

        soft_target = _tgca_soft_branch_target_v3(branch_oracle, device=y_pred.device)
        branch_loss = F.kl_div(branch_probs.log(), soft_target, reduction="batchmean")

        oracle_boost_rate = (branch_oracle == 0).float().mean()
        oracle_damp_rate = (branch_oracle == 1).float().mean()
        oracle_fallback_rate = (branch_oracle == 2).float().mean()
        oracle_neutral_rate = (branch_oracle == 3).float().mean()

    curvature_proxy = aux.get("curvature_proxy", torch.zeros_like(alpha))
    curv_loss = F.relu(curvature_proxy - 5.0).mean()

    try:
        seg_ratio = segment_energy_ratio(r_base.detach(), r_final, n_segments=3, eps=eps)
        segment_loss = F.relu(seg_ratio).mean()
    except Exception:
        segment_loss = torch.zeros((), device=y_pred.device)

    loss = (
        forecast_loss
        + lambda_energy * energy_hinge
        + lambda_state * state_loss
        + lambda_spec * spec_js
        + lambda_tail * tail_loss
        + lambda_alpha * alpha_loss
        + lambda_branch * branch_loss
        + lambda_curv * curv_loss
        + lambda_segment * segment_loss
    )

    with torch.no_grad():
        branch_probs = aux.get("branch_probs", torch.zeros(y_pred.size(0), 4, device=y_pred.device))
        spectral_bad = aux.get("spectral_bad", torch.zeros_like(alpha))
        direction_cos = aux.get("direction_cos", torch.zeros_like(alpha))
        severe_direction_bad = aux.get("severe_direction_bad", torch.zeros_like(alpha))
        alpha_branch_eff = aux.get("alpha_branch_eff", torch.zeros_like(alpha))

        logs = {
            "loss": float(loss.detach().cpu()),
            "mse": float(mse.detach().cpu()),
            "mae": float(mae.detach().cpu()),
            "base_mse": float(F.mse_loss(y0, y_true).detach().cpu()),
            "base_mae": float(F.l1_loss(y0, y_true).detach().cpu()),
            "energy_hinge": float(energy_hinge.detach().cpu()),
            "state_loss": float(state_loss.detach().cpu()),
            "spec_js": float(spec_js.detach().cpu()),
            "tail_base": float(tail_base.detach().cpu()),
            "tail_final": float(tail_final.detach().cpu()),
            "alpha_mean": float(alpha.mean().detach().cpu()),
            "alpha_max": float(alpha.max().detach().cpu()),
            "alpha_branch_eff": float(alpha_branch_eff.mean().detach().cpu()),
            "boost_rate": float(branch_probs[:, 0].mean().detach().cpu()) if branch_probs.ndim == 2 and branch_probs.size(1) > 0 else 0.0,
            "damp_rate": float(branch_probs[:, 1].mean().detach().cpu()) if branch_probs.ndim == 2 and branch_probs.size(1) > 1 else 0.0,
            "fallback_rate": float(branch_probs[:, 2].mean().detach().cpu()) if branch_probs.ndim == 2 and branch_probs.size(1) > 2 else 0.0,
            "neutral_rate": float(branch_probs[:, 3].mean().detach().cpu()) if branch_probs.ndim == 2 and branch_probs.size(1) > 3 else 0.0,
            "spectral_bad_rate": float(spectral_bad.mean().detach().cpu()),
            "branch_loss": float(branch_loss.detach().cpu()),
            "curv_loss": float(curv_loss.detach().cpu()),
            "segment_loss": float(segment_loss.detach().cpu()),
            "curvature_proxy": float(curvature_proxy.mean().detach().cpu()),
            "direction_cos": float(direction_cos.mean().detach().cpu()),
            "severe_direction_bad_rate": float(severe_direction_bad.mean().detach().cpu()),
            "oracle_boost_rate": float(oracle_boost_rate.detach().cpu()),
            "oracle_damp_rate": float(oracle_damp_rate.detach().cpu()),
            "oracle_fallback_rate": float(oracle_fallback_rate.detach().cpu()),
            "oracle_neutral_rate": float(oracle_neutral_rate.detach().cpu()),
        }

    return loss, logs


# Rebind projector and forward. Keep forward from V2 if available because it already injects branch_oracle.
HardTrustProjector = HardTrustProjectorV3

if "_tgca_forward_v2" in globals():
    TGCAForecaster.forward = _tgca_forward_v2

# ===== TGCA_V3_SOFT_REAL_END =====




# ===== TGCA_V4_DECOUPLED_START =====
# V4:
#   Keep v1 strong correction.
#   Keep v3 diagnostic mechanisms.
#   Decouple branch diagnostic supervision from alpha control.

class HardTrustProjectorV4(nn.Module):
    """
    Decoupled trust projector.

    alpha is controlled by:
      - alpha_raw
      - alpha_energy
      - spectral hard gate
      - severe curvature/direction risk

    branch router is diagnostic, not a direct alpha bottleneck.
    """

    def __init__(self, cfg: TGCAConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        y0: torch.Tensor,
        delta_raw: torch.Tensor,
        r_proxy: torch.Tensor,
        alpha_raw: torch.Tensor,
        alpha_branch: torch.Tensor,
        y_true_norm: Optional[torch.Tensor] = None,
    ):
        eps = self.cfg.eps
        r_ref = y0 - y_true_norm if y_true_norm is not None else r_proxy

        inner = (r_ref * delta_raw).mean(dim=(1, 2), keepdim=True)
        delta_energy = delta_raw.pow(2).mean(dim=(1, 2), keepdim=True) + eps
        residual_energy = r_ref.pow(2).mean(dim=(1, 2), keepdim=True) + eps

        alpha_energy = (-2.0 * inner / delta_energy).clamp(0.0, self.cfg.alpha_max)

        if y_true_norm is None:
            z = torch.zeros_like(alpha_raw)
            alpha_raw_scaled = self.cfg.alpha_max * alpha_raw
            alpha_train_cap = torch.full_like(alpha_raw, min(0.15, self.cfg.alpha_max))
            alpha = torch.minimum(alpha_raw_scaled, alpha_train_cap)
            y = y0 + alpha * delta_raw

            return y, {
                "alpha": alpha,
                "alpha_raw": alpha_raw_scaled,
                "alpha_energy": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "alpha_branch": alpha_branch,
                "alpha_branch_eff": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "alpha_spec": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "alpha_curv": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "alpha_dir": torch.full_like(alpha_raw, self.cfg.alpha_max),
                "js_raw": z,
                "kl_raw": z,
                "rkl_raw": z,
                "kl_asym_raw": z,
                "spectral_bad": z,
                "curvature_proxy": z,
                "direction_cos": z,
                "severe_direction_bad": z,
            }

        p0 = spectral_prob(r_ref, eps=eps)
        p1 = spectral_prob(r_ref + delta_raw, eps=eps)

        js = js_div(p0, p1, eps=eps).view(-1, 1, 1)
        kl01 = kl_div(p0, p1, eps=eps).view(-1, 1, 1)
        kl10 = kl_div(p1, p0, eps=eps).view(-1, 1, 1)
        kl_asym = kl01 - kl10

        spectral_bad = (
            (js > self.cfg.js_limit)
            | ((kl01 > kl10) & (kl_asym > 0.05) & (js > 0.05))
        )

        alpha_spec = torch.where(
            spectral_bad,
            torch.full_like(alpha_raw, self.cfg.alpha_fallback),
            torch.full_like(alpha_raw, self.cfg.alpha_max),
        )

        curvature_proxy = delta_energy / residual_energy

        # Only extreme curvature risk caps alpha.
        alpha_curv = torch.where(
            curvature_proxy > 8.0,
            torch.full_like(alpha_raw, min(self.cfg.alpha_damp_cap, self.cfg.alpha_max)),
            torch.full_like(alpha_raw, self.cfg.alpha_max),
        )

        if "_tgca_safe_cosine_v2" in globals():
            direction_cos = _tgca_safe_cosine_v2(r_ref, delta_raw, eps=eps)
        elif "safe_cosine" in globals():
            direction_cos = safe_cosine(r_ref, delta_raw, eps=eps)
        else:
            num = (r_ref * delta_raw).sum(dim=(1, 2), keepdim=True)
            den = torch.sqrt(r_ref.pow(2).sum(dim=(1, 2), keepdim=True) + eps) * torch.sqrt(delta_raw.pow(2).sum(dim=(1, 2), keepdim=True) + eps)
            direction_cos = num / den.clamp_min(eps)

        # Only severe wrong direction caps alpha.
        severe_direction_bad = (direction_cos > 0.50) & (alpha_energy < 0.03 * self.cfg.alpha_max)

        alpha_dir = torch.where(
            severe_direction_bad,
            torch.full_like(alpha_raw, min(0.50 * self.cfg.alpha_max, self.cfg.alpha_max)),
            torch.full_like(alpha_raw, self.cfg.alpha_max),
        )

        alpha_raw_scaled = self.cfg.alpha_max * alpha_raw

        # Key difference: no alpha_branch bottleneck.
        alpha_branch_eff = torch.full_like(alpha_raw, self.cfg.alpha_max)

        alpha = torch.minimum(alpha_raw_scaled, alpha_energy)
        alpha = torch.minimum(alpha, alpha_spec)
        alpha = torch.minimum(alpha, alpha_curv)
        alpha = torch.minimum(alpha, alpha_dir)

        y = y0 + alpha * delta_raw

        return y, {
            "alpha": alpha,
            "alpha_raw": alpha_raw_scaled,
            "alpha_energy": alpha_energy,
            "alpha_branch": alpha_branch,
            "alpha_branch_eff": alpha_branch_eff,
            "alpha_spec": alpha_spec,
            "alpha_curv": alpha_curv,
            "alpha_dir": alpha_dir,
            "js_raw": js,
            "kl_raw": kl01,
            "rkl_raw": kl10,
            "kl_asym_raw": kl_asym,
            "spectral_bad": spectral_bad.float(),
            "curvature_proxy": curvature_proxy,
            "direction_cos": direction_cos,
            "severe_direction_bad": severe_direction_bad.float(),
        }

HardTrustProjector = HardTrustProjectorV4

if "_tgca_forward_v2" in globals():
    TGCAForecaster.forward = _tgca_forward_v2

# ===== TGCA_V4_DECOUPLED_END =====




# ===== TGCA_V5_DETACHED_DIAG_START =====
# V5:
#   Keep V4 decoupled predictor.
#   Keep V3 soft diagnostic branch loss.
#   But detach diagnostic branch input, so branch_loss trains branch_head only
#   and does not perturb shared state z used by op_probs / alpha_raw / correction bank.

def _tgca_forward_v5_detached_diag(
    self,
    x_enc: torch.Tensor,
    x_mark_enc: Optional[torch.Tensor] = None,
    x_dec: Optional[torch.Tensor] = None,
    x_mark_dec: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    y_true: Optional[torch.Tensor] = None,
    return_aux: bool = True,
):
    x_norm, mean, std = self._norm(x_enc)

    y0_norm = self.base(x_norm)

    if y_true is not None:
        y_true_norm = (y_true - mean) / std
    else:
        y_true_norm = None

    state = self.state(x_norm, y0_norm)

    # Prediction path: op_probs and alpha_raw keep full gradients through z.
    # The branch logits here are irrelevant for V4/V5 alpha because branch is not an alpha bottleneck.
    branch_probs_pred, op_probs, alpha_branch = self.router(
        state["branch_logits"],
        state["op_logits"],
    )

    # Diagnostic path: branch head receives detached z.
    # This lets branch_loss train branch_head, but prevents it from changing z,
    # op_probs, alpha_raw, or correction operators.
    if hasattr(self.state, "branch_head"):
        branch_logits_diag = self.state.branch_head(state["z"].detach())
        branch_probs_diag = torch.softmax(branch_logits_diag / self.cfg.router_tau, dim=-1)
    else:
        branch_probs_diag = branch_probs_pred.detach()

    deltas = self.bank(y0_norm, state["z"])
    delta_raw = (deltas * op_probs[:, :, None, None]).sum(dim=1)

    y_norm, trust_info = self.projector(
        y0=y0_norm,
        delta_raw=delta_raw,
        r_proxy=state["r_proxy"],
        alpha_raw=state["alpha_raw"],
        alpha_branch=alpha_branch,
        y_true_norm=y_true_norm,
    )

    if y_true_norm is not None:
        r_base_norm = y0_norm - y_true_norm

        if "_tgca_make_branch_oracle_v2" in globals():
            trust_info["branch_oracle"] = _tgca_make_branch_oracle_v2(
                r_base=r_base_norm,
                delta_raw=delta_raw,
                alpha_energy=trust_info["alpha_energy"],
                js_raw=trust_info["js_raw"],
                kl_raw=trust_info["kl_raw"],
                rkl_raw=trust_info["rkl_raw"],
                kl_asym_raw=trust_info["kl_asym_raw"],
                cfg=self.cfg,
            )
        elif "make_tgca_branch_oracle" in globals():
            trust_info["branch_oracle"] = make_tgca_branch_oracle(
                r_base=r_base_norm,
                delta_raw=delta_raw,
                alpha_energy=trust_info["alpha_energy"],
                js_raw=trust_info["js_raw"],
                kl_raw=trust_info["kl_raw"],
                rkl_raw=trust_info["rkl_raw"],
                kl_asym_raw=trust_info["kl_asym_raw"],
                cfg=self.cfg,
            )

    y = self._denorm(y_norm, mean, std)
    y0 = self._denorm(y0_norm, mean, std)
    r_proxy = state["r_proxy"] * std

    if not return_aux:
        return y

    aux = {
        "y0": y0,
        "y_norm": y_norm,
        "y0_norm": y0_norm,
        "delta_raw_norm": delta_raw,
        "r_proxy": r_proxy,
        "r_proxy_norm": state["r_proxy"],

        # Important:
        # branch_probs used by loss/logs are diagnostic detached-encoder branch probs.
        "branch_probs": branch_probs_diag,

        # Keep prediction branch probs for debugging.
        "branch_probs_pred": branch_probs_pred,

        "op_probs": op_probs,
        "deltas_norm": deltas,
        "mean": mean,
        "std": std,
    }

    aux.update(trust_info)
    return y, aux


# Keep V4 projector if available. V4 is the best trust projector so far.
if "HardTrustProjectorV4" in globals():
    HardTrustProjector = HardTrustProjectorV4

TGCAForecaster.forward = _tgca_forward_v5_detached_diag

# ===== TGCA_V5_DETACHED_DIAG_END =====




# ===== TGCA_V7A_SEGMENT_TAIL_START =====
# V7A:
#   1. Segment-wise correction for PL720: front / middle / back.
#   2. Segment-wise op_probs and alpha.
#   3. Tail energy-ratio + tail correlation alignment loss.
#   4. Keep v5-style detached diagnostic branch.

def _tgca_segment_slices_v7(h: int, n_segments: int = 3):
    return [(int(i * h / n_segments), int((i + 1) * h / n_segments)) for i in range(n_segments)]


def _tgca_tail_corr_v7(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Batch-wise flattened cosine/correlation on tail segment.
    Return [B].
    """
    af = a.reshape(a.size(0), -1)
    bf = b.reshape(b.size(0), -1)
    num = (af * bf).sum(dim=-1)
    den = torch.sqrt(af.pow(2).sum(dim=-1) + eps) * torch.sqrt(bf.pow(2).sum(dim=-1) + eps)
    return num / den.clamp_min(eps)


def _tgca_tail_alignment_loss_v7(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y0: torch.Tensor,
    tail_ratio: float = 1.0 / 3.0,
    eps: float = 1e-6,
):
    """
    Tail loss from the document logic:
      tail MSE improves when energy ratio rho_tail -> 1 and Corr_tail increases.

    L = |log rho_tail| + ReLU(Corr_base_tail - Corr_pred_tail)
    """
    h = y_true.size(1)
    start = int((1.0 - tail_ratio) * h)

    yp = y_pred[:, start:, :]
    yt = y_true[:, start:, :]
    yb = y0[:, start:, :]

    e_pred = yp.pow(2).mean(dim=(1, 2))
    e_true = yt.pow(2).mean(dim=(1, 2))
    rho = (e_pred + eps) / (e_true + eps)

    corr_pred = _tgca_tail_corr_v7(yp, yt, eps=eps)
    corr_base = _tgca_tail_corr_v7(yb, yt, eps=eps)

    energy_loss = torch.abs(torch.log(rho + eps)).mean()
    corr_loss = F.relu(corr_base - corr_pred).mean()

    return energy_loss + corr_loss, {
        "tail_energy_ratio": rho.mean(),
        "tail_corr_pred": corr_pred.mean(),
        "tail_corr_base": corr_base.mean(),
        "tail_energy_loss": energy_loss,
        "tail_corr_loss": corr_loss,
    }


# Preserve the current TGCAForecaster.__init__, then add segment heads before optimizer is built.
if not hasattr(TGCAForecaster, "_tgca_v7a_old_init"):
    TGCAForecaster._tgca_v7a_old_init = TGCAForecaster.__init__


def _tgca_init_v7a(self, cfg: TGCAConfig):
    TGCAForecaster._tgca_v7a_old_init(self, cfg)

    # Segment-wise operator and alpha heads.
    # They are created in __init__, so optimizer will see them.
    self.seg_op_head = nn.Linear(cfg.d_state, cfg.n_segments * cfg.n_ops)
    self.seg_alpha_head = nn.Linear(cfg.d_state, cfg.n_segments)

    nn.init.zeros_(self.seg_op_head.weight)
    nn.init.zeros_(self.seg_op_head.bias)
    nn.init.zeros_(self.seg_alpha_head.weight)
    nn.init.zeros_(self.seg_alpha_head.bias)


class HardTrustProjectorV7A(nn.Module):
    """
    Segment-wise trust projector.

    alpha_seg is controlled per segment by:
      - segment alpha raw
      - segment energy-safe alpha
      - spectral hard gate
      - severe curvature/direction safety

    Branch diagnostic does not bottleneck alpha.
    """

    def __init__(self, cfg: TGCAConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        y0: torch.Tensor,
        delta_raw: torch.Tensor,
        r_proxy: torch.Tensor,
        alpha_raw: torch.Tensor,
        alpha_branch: torch.Tensor,
        y_true_norm: Optional[torch.Tensor] = None,
    ):
        eps = self.cfg.eps
        b, h, c = y0.shape
        s_num = self.cfg.n_segments
        slices = _tgca_segment_slices_v7(h, s_num)

        r_ref = y0 - y_true_norm if y_true_norm is not None else r_proxy

        # alpha_raw may be [B,S,1] in V7A.
        if alpha_raw.dim() == 3 and alpha_raw.size(1) == s_num:
            alpha_raw_seg = alpha_raw
        else:
            alpha_raw_seg = alpha_raw.view(b, 1, 1).expand(b, s_num, 1)

        alpha_raw_scaled = self.cfg.alpha_max * alpha_raw_seg

        # No-y_true path: small trainable segment correction.
        if y_true_norm is None:
            alpha_train_cap = torch.full_like(alpha_raw_scaled, min(0.15, self.cfg.alpha_max))
            alpha_seg = torch.minimum(alpha_raw_scaled, alpha_train_cap)

            alpha_map = torch.zeros(b, h, 1, device=y0.device, dtype=y0.dtype)
            for i, (st, ed) in enumerate(slices):
                alpha_map[:, st:ed, :] = alpha_seg[:, i:i+1, :].view(b, 1, 1)

            y = y0 + alpha_map * delta_raw
            z = torch.zeros(b, 1, 1, device=y0.device, dtype=y0.dtype)

            return y, {
                "alpha": alpha_seg.mean(dim=1, keepdim=True),
                "alpha_seg": alpha_seg.squeeze(-1),
                "alpha_raw": alpha_raw_scaled.mean(dim=1, keepdim=True),
                "alpha_energy": torch.full((b, 1, 1), self.cfg.alpha_max, device=y0.device, dtype=y0.dtype),
                "alpha_branch": alpha_branch,
                "alpha_branch_eff": torch.full((b, 1, 1), self.cfg.alpha_max, device=y0.device, dtype=y0.dtype),
                "alpha_spec": torch.full((b, 1, 1), self.cfg.alpha_max, device=y0.device, dtype=y0.dtype),
                "alpha_curv": torch.full((b, 1, 1), self.cfg.alpha_max, device=y0.device, dtype=y0.dtype),
                "alpha_dir": torch.full((b, 1, 1), self.cfg.alpha_max, device=y0.device, dtype=y0.dtype),
                "js_raw": z,
                "kl_raw": z,
                "rkl_raw": z,
                "kl_asym_raw": z,
                "spectral_bad": z,
                "curvature_proxy": z,
                "direction_cos": z,
                "severe_direction_bad": z,
            }

        # Global spectral diagnostics.
        p0 = spectral_prob(r_ref, eps=eps)
        p1 = spectral_prob(r_ref + delta_raw, eps=eps)

        js = js_div(p0, p1, eps=eps).view(-1, 1, 1)
        kl01 = kl_div(p0, p1, eps=eps).view(-1, 1, 1)
        kl10 = kl_div(p1, p0, eps=eps).view(-1, 1, 1)
        kl_asym = kl01 - kl10

        spectral_bad = (
            (js > self.cfg.js_limit)
            | ((kl01 > kl10) & (kl_asym > 0.05) & (js > 0.05))
        )

        alpha_spec = torch.where(
            spectral_bad,
            torch.full((b, 1, 1), self.cfg.alpha_fallback, device=y0.device, dtype=y0.dtype),
            torch.full((b, 1, 1), self.cfg.alpha_max, device=y0.device, dtype=y0.dtype),
        )

        # Segment-wise alpha energy.
        alpha_energy_seg = []
        curvature_seg = []
        direction_seg = []
        severe_dir_seg = []

        for st, ed in slices:
            rr = r_ref[:, st:ed, :]
            dd = delta_raw[:, st:ed, :]

            inner = (rr * dd).mean(dim=(1, 2), keepdim=True)
            d_energy = dd.pow(2).mean(dim=(1, 2), keepdim=True) + eps
            r_energy = rr.pow(2).mean(dim=(1, 2), keepdim=True) + eps

            ae = (-2.0 * inner / d_energy).clamp(0.0, self.cfg.alpha_max)

            curv = d_energy / r_energy

            if "_tgca_safe_cosine_v2" in globals():
                dc = _tgca_safe_cosine_v2(rr, dd, eps=eps)
            elif "safe_cosine" in globals():
                dc = safe_cosine(rr, dd, eps=eps)
            else:
                num = (rr * dd).sum(dim=(1, 2), keepdim=True)
                den = torch.sqrt(rr.pow(2).sum(dim=(1, 2), keepdim=True) + eps) * torch.sqrt(dd.pow(2).sum(dim=(1, 2), keepdim=True) + eps)
                dc = num / den.clamp_min(eps)

            sdb = (dc > 0.50) & (ae < 0.03 * self.cfg.alpha_max)

            alpha_energy_seg.append(ae)
            curvature_seg.append(curv)
            direction_seg.append(dc)
            severe_dir_seg.append(sdb.float())

        alpha_energy_seg = torch.cat(alpha_energy_seg, dim=1)      # [B,S,1]
        curvature_proxy_seg = torch.cat(curvature_seg, dim=1)     # [B,S,1]
        direction_cos_seg = torch.cat(direction_seg, dim=1)       # [B,S,1]
        severe_direction_bad_seg = torch.cat(severe_dir_seg, dim=1)

        alpha_curv_seg = torch.where(
            curvature_proxy_seg > 8.0,
            torch.full_like(alpha_raw_scaled, min(self.cfg.alpha_damp_cap, self.cfg.alpha_max)),
            torch.full_like(alpha_raw_scaled, self.cfg.alpha_max),
        )

        alpha_dir_seg = torch.where(
            severe_direction_bad_seg > 0,
            torch.full_like(alpha_raw_scaled, min(0.50 * self.cfg.alpha_max, self.cfg.alpha_max)),
            torch.full_like(alpha_raw_scaled, self.cfg.alpha_max),
        )

        alpha_spec_seg = alpha_spec.expand(b, s_num, 1)

        alpha_seg = torch.minimum(alpha_raw_scaled, alpha_energy_seg)
        alpha_seg = torch.minimum(alpha_seg, alpha_spec_seg)
        alpha_seg = torch.minimum(alpha_seg, alpha_curv_seg)
        alpha_seg = torch.minimum(alpha_seg, alpha_dir_seg)

        alpha_map = torch.zeros(b, h, 1, device=y0.device, dtype=y0.dtype)
        for i, (st, ed) in enumerate(slices):
            alpha_map[:, st:ed, :] = alpha_seg[:, i:i+1, :].view(b, 1, 1)

        y = y0 + alpha_map * delta_raw

        return y, {
            "alpha": alpha_seg.mean(dim=1, keepdim=True),
            "alpha_seg": alpha_seg.squeeze(-1),
            "alpha_raw": alpha_raw_scaled.mean(dim=1, keepdim=True),
            "alpha_raw_seg": alpha_raw_scaled.squeeze(-1),
            "alpha_energy": alpha_energy_seg.mean(dim=1, keepdim=True),
            "alpha_energy_seg": alpha_energy_seg.squeeze(-1),
            "alpha_branch": alpha_branch,
            "alpha_branch_eff": torch.full((b, 1, 1), self.cfg.alpha_max, device=y0.device, dtype=y0.dtype),
            "alpha_spec": alpha_spec,
            "alpha_curv": alpha_curv_seg.mean(dim=1, keepdim=True),
            "alpha_dir": alpha_dir_seg.mean(dim=1, keepdim=True),
            "js_raw": js,
            "kl_raw": kl01,
            "rkl_raw": kl10,
            "kl_asym_raw": kl_asym,
            "spectral_bad": spectral_bad.float(),
            "curvature_proxy": curvature_proxy_seg.mean(dim=1, keepdim=True),
            "curvature_proxy_seg": curvature_proxy_seg.squeeze(-1),
            "direction_cos": direction_cos_seg.mean(dim=1, keepdim=True),
            "direction_cos_seg": direction_cos_seg.squeeze(-1),
            "severe_direction_bad": severe_direction_bad_seg.mean(dim=1, keepdim=True),
        }


def _tgca_forward_v7a_segment_tail(
    self,
    x_enc: torch.Tensor,
    x_mark_enc: Optional[torch.Tensor] = None,
    x_dec: Optional[torch.Tensor] = None,
    x_mark_dec: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    y_true: Optional[torch.Tensor] = None,
    return_aux: bool = True,
):
    x_norm, mean, std = self._norm(x_enc)
    y0_norm = self.base(x_norm)

    y_true_norm = (y_true - mean) / std if y_true is not None else None

    state = self.state(x_norm, y0_norm)

    # Global diagnostic branch still exists, detached from shared z.
    branch_probs_pred, op_probs_global, alpha_branch = self.router(
        state["branch_logits"],
        state["op_logits"],
    )

    if hasattr(self.state, "branch_head"):
        branch_logits_diag = self.state.branch_head(state["z"].detach())
        branch_probs_diag = torch.softmax(branch_logits_diag / self.cfg.router_tau, dim=-1)
    else:
        branch_probs_diag = branch_probs_pred.detach()

    deltas = self.bank(y0_norm, state["z"])  # [B, n_ops, H, C]
    b, _, h, c = deltas.shape
    s_num = self.cfg.n_segments
    slices = _tgca_segment_slices_v7(h, s_num)

    if hasattr(self, "seg_op_head"):
        seg_op_logits = self.seg_op_head(state["z"]).view(b, s_num, self.cfg.n_ops)
        seg_op_probs = torch.softmax(seg_op_logits / self.cfg.router_tau, dim=-1)
    else:
        seg_op_probs = op_probs_global[:, None, :].expand(b, s_num, self.cfg.n_ops)

    delta_raw = torch.zeros(b, h, c, device=x_enc.device, dtype=x_enc.dtype)
    for i, (st, ed) in enumerate(slices):
        w = seg_op_probs[:, i, :, None, None]
        delta_raw[:, st:ed, :] = (deltas[:, :, st:ed, :] * w).sum(dim=1)

    if hasattr(self, "seg_alpha_head"):
        alpha_raw_seg = torch.sigmoid(self.seg_alpha_head(state["z"])).view(b, s_num, 1)
    else:
        alpha_raw_seg = state["alpha_raw"].view(b, 1, 1).expand(b, s_num, 1)

    y_norm, trust_info = self.projector(
        y0=y0_norm,
        delta_raw=delta_raw,
        r_proxy=state["r_proxy"],
        alpha_raw=alpha_raw_seg,
        alpha_branch=alpha_branch,
        y_true_norm=y_true_norm,
    )

    if y_true_norm is not None:
        r_base_norm = y0_norm - y_true_norm

        if "_tgca_make_branch_oracle_v2" in globals():
            trust_info["branch_oracle"] = _tgca_make_branch_oracle_v2(
                r_base=r_base_norm,
                delta_raw=delta_raw,
                alpha_energy=trust_info["alpha_energy"],
                js_raw=trust_info["js_raw"],
                kl_raw=trust_info["kl_raw"],
                rkl_raw=trust_info["rkl_raw"],
                kl_asym_raw=trust_info["kl_asym_raw"],
                cfg=self.cfg,
            )
        elif "make_tgca_branch_oracle" in globals():
            trust_info["branch_oracle"] = make_tgca_branch_oracle(
                r_base=r_base_norm,
                delta_raw=delta_raw,
                alpha_energy=trust_info["alpha_energy"],
                js_raw=trust_info["js_raw"],
                kl_raw=trust_info["kl_raw"],
                rkl_raw=trust_info["rkl_raw"],
                kl_asym_raw=trust_info["kl_asym_raw"],
                cfg=self.cfg,
            )

    y = self._denorm(y_norm, mean, std)
    y0 = self._denorm(y0_norm, mean, std)
    r_proxy = state["r_proxy"] * std

    if not return_aux:
        return y

    aux = {
        "y0": y0,
        "y_norm": y_norm,
        "y0_norm": y0_norm,
        "delta_raw_norm": delta_raw,
        "r_proxy": r_proxy,
        "r_proxy_norm": state["r_proxy"],
        "branch_probs": branch_probs_diag,
        "branch_probs_pred": branch_probs_pred,
        "op_probs": op_probs_global,
        "seg_op_probs": seg_op_probs,
        "deltas_norm": deltas,
        "mean": mean,
        "std": std,
    }

    aux.update(trust_info)
    return y, aux


def tgca_forecasting_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    beta_mae: float = 1.0,
    lambda_energy: float = 0.20,
    lambda_state: float = 0.05,
    lambda_spec: float = 0.05,
    lambda_tail: float = 0.05,
    lambda_alpha: float = 0.001,
    lambda_branch: float = 0.02,
    lambda_curv: float = 0.02,
    lambda_segment: float = 0.05,
    lambda_tail_align: float = 0.02,
    margin: float = 0.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    y0 = aux["y0"]
    r_base = y0 - y_true
    r_final = y_pred - y_true

    mse = F.mse_loss(y_pred, y_true)
    mae = F.l1_loss(y_pred, y_true)
    forecast_loss = mse + beta_mae * mae

    base_energy = r_base.pow(2).mean()
    final_energy = r_final.pow(2).mean()
    energy_hinge = F.relu(torch.log(final_energy + eps) - torch.log(base_energy + eps) + margin)

    r_proxy = aux.get("r_proxy", torch.zeros_like(r_base))
    state_loss = F.smooth_l1_loss(r_proxy, r_base.detach())

    p_base = spectral_prob(r_base.detach(), eps=eps)
    p_final = spectral_prob(r_final, eps=eps)
    spec_js = js_div(p_base, p_final, eps=eps).mean()

    tail_base = boltzmann_tail_mass(r_base.detach(), eps=eps)
    tail_final = boltzmann_tail_mass(r_final, eps=eps)
    tail_loss = F.relu(tail_final - tail_base)

    tail_align_loss, tail_logs = _tgca_tail_alignment_loss_v7(
        y_pred=y_pred,
        y_true=y_true,
        y0=y0,
        tail_ratio=1.0 / 3.0,
        eps=eps,
    )

    alpha = aux.get("alpha", torch.zeros(y_pred.size(0), 1, 1, device=y_pred.device))
    alpha_loss = alpha.pow(2).mean()

    branch_loss = torch.zeros((), device=y_pred.device)
    oracle_boost_rate = torch.zeros((), device=y_pred.device)
    oracle_damp_rate = torch.zeros((), device=y_pred.device)
    oracle_fallback_rate = torch.zeros((), device=y_pred.device)
    oracle_neutral_rate = torch.zeros((), device=y_pred.device)

    if "branch_oracle" in aux and "branch_probs" in aux:
        branch_oracle = aux["branch_oracle"].long()
        branch_probs = aux["branch_probs"].clamp_min(eps)

        if "_tgca_soft_branch_target_v3" in globals():
            soft_target = _tgca_soft_branch_target_v3(branch_oracle, device=y_pred.device)
            branch_loss = F.kl_div(branch_probs.log(), soft_target, reduction="batchmean")
        else:
            branch_loss = F.nll_loss(branch_probs.log(), branch_oracle)

        oracle_boost_rate = (branch_oracle == 0).float().mean()
        oracle_damp_rate = (branch_oracle == 1).float().mean()
        oracle_fallback_rate = (branch_oracle == 2).float().mean()
        oracle_neutral_rate = (branch_oracle == 3).float().mean()

    curvature_proxy = aux.get("curvature_proxy", torch.zeros_like(alpha))
    curv_loss = F.relu(curvature_proxy - 8.0).mean()

    try:
        seg_ratio = segment_energy_ratio(r_base.detach(), r_final, n_segments=3, eps=eps)
        segment_loss = F.relu(seg_ratio).mean()
    except Exception:
        segment_loss = torch.zeros((), device=y_pred.device)

    loss = (
        forecast_loss
        + lambda_energy * energy_hinge
        + lambda_state * state_loss
        + lambda_spec * spec_js
        + lambda_tail * tail_loss
        + lambda_alpha * alpha_loss
        + lambda_branch * branch_loss
        + lambda_curv * curv_loss
        + lambda_segment * segment_loss
        + lambda_tail_align * tail_align_loss
    )

    with torch.no_grad():
        branch_probs = aux.get("branch_probs", torch.zeros(y_pred.size(0), 4, device=y_pred.device))
        spectral_bad = aux.get("spectral_bad", torch.zeros_like(alpha))
        direction_cos = aux.get("direction_cos", torch.zeros_like(alpha))
        severe_direction_bad = aux.get("severe_direction_bad", torch.zeros_like(alpha))
        alpha_branch_eff = aux.get("alpha_branch_eff", torch.zeros_like(alpha))
        alpha_seg = aux.get("alpha_seg", torch.zeros(y_pred.size(0), 3, device=y_pred.device))
        seg_op_probs = aux.get("seg_op_probs", torch.zeros(y_pred.size(0), 3, 1, device=y_pred.device))

        logs = {
            "loss": float(loss.detach().cpu()),
            "mse": float(mse.detach().cpu()),
            "mae": float(mae.detach().cpu()),
            "base_mse": float(F.mse_loss(y0, y_true).detach().cpu()),
            "base_mae": float(F.l1_loss(y0, y_true).detach().cpu()),
            "energy_hinge": float(energy_hinge.detach().cpu()),
            "state_loss": float(state_loss.detach().cpu()),
            "spec_js": float(spec_js.detach().cpu()),
            "tail_base": float(tail_base.detach().cpu()),
            "tail_final": float(tail_final.detach().cpu()),
            "tail_align_loss": float(tail_align_loss.detach().cpu()),
            "tail_energy_ratio": float(tail_logs["tail_energy_ratio"].detach().cpu()),
            "tail_corr_pred": float(tail_logs["tail_corr_pred"].detach().cpu()),
            "tail_corr_base": float(tail_logs["tail_corr_base"].detach().cpu()),
            "tail_energy_loss": float(tail_logs["tail_energy_loss"].detach().cpu()),
            "tail_corr_loss": float(tail_logs["tail_corr_loss"].detach().cpu()),
            "alpha_mean": float(alpha.mean().detach().cpu()),
            "alpha_max": float(alpha.max().detach().cpu()),
            "alpha_front": float(alpha_seg[:, 0].mean().detach().cpu()) if alpha_seg.ndim == 2 and alpha_seg.size(1) > 0 else 0.0,
            "alpha_mid": float(alpha_seg[:, 1].mean().detach().cpu()) if alpha_seg.ndim == 2 and alpha_seg.size(1) > 1 else 0.0,
            "alpha_back": float(alpha_seg[:, 2].mean().detach().cpu()) if alpha_seg.ndim == 2 and alpha_seg.size(1) > 2 else 0.0,
            "alpha_branch_eff": float(alpha_branch_eff.mean().detach().cpu()),
            "boost_rate": float(branch_probs[:, 0].mean().detach().cpu()) if branch_probs.ndim == 2 and branch_probs.size(1) > 0 else 0.0,
            "damp_rate": float(branch_probs[:, 1].mean().detach().cpu()) if branch_probs.ndim == 2 and branch_probs.size(1) > 1 else 0.0,
            "fallback_rate": float(branch_probs[:, 2].mean().detach().cpu()) if branch_probs.ndim == 2 and branch_probs.size(1) > 2 else 0.0,
            "neutral_rate": float(branch_probs[:, 3].mean().detach().cpu()) if branch_probs.ndim == 2 and branch_probs.size(1) > 3 else 0.0,
            "spectral_bad_rate": float(spectral_bad.mean().detach().cpu()),
            "branch_loss": float(branch_loss.detach().cpu()),
            "curv_loss": float(curv_loss.detach().cpu()),
            "segment_loss": float(segment_loss.detach().cpu()),
            "curvature_proxy": float(curvature_proxy.mean().detach().cpu()),
            "direction_cos": float(direction_cos.mean().detach().cpu()),
            "severe_direction_bad_rate": float(severe_direction_bad.mean().detach().cpu()),
            "oracle_boost_rate": float(oracle_boost_rate.detach().cpu()),
            "oracle_damp_rate": float(oracle_damp_rate.detach().cpu()),
            "oracle_fallback_rate": float(oracle_fallback_rate.detach().cpu()),
            "oracle_neutral_rate": float(oracle_neutral_rate.detach().cpu()),
            "seg_op_front_max": float(seg_op_probs[:, 0, :].max(dim=-1).values.mean().detach().cpu()) if seg_op_probs.ndim == 3 and seg_op_probs.size(1) > 0 else 0.0,
            "seg_op_mid_max": float(seg_op_probs[:, 1, :].max(dim=-1).values.mean().detach().cpu()) if seg_op_probs.ndim == 3 and seg_op_probs.size(1) > 1 else 0.0,
            "seg_op_back_max": float(seg_op_probs[:, 2, :].max(dim=-1).values.mean().detach().cpu()) if seg_op_probs.ndim == 3 and seg_op_probs.size(1) > 2 else 0.0,
        }

    return loss, logs


TGCAForecaster.__init__ = _tgca_init_v7a
HardTrustProjector = HardTrustProjectorV7A
TGCAForecaster.forward = _tgca_forward_v7a_segment_tail

# ===== TGCA_V7A_SEGMENT_TAIL_END =====




# ===== TGCA_V8A_BOLTZMANN_SEGMENT_PRIOR_START =====
# V8A:
#   Mechanism from the uploaded documents:
#   Boltzmann hard-mode segment prior.
#
#   Document logic:
#     h_i = normalized hard energy
#     b_i(T) = exp(h_i / T) / Z_T
#     BTM_tail(T) = sum_{i in tail} b_i(T)
#
#   Model use:
#     compute segment hard energy from r_proxy
#     use Boltzmann mass to softly reweight alpha_front/mid/back
#     no extra loss; this is structural prior, not another penalty.

def _tgca_segment_hard_boltzmann_v8(
    r_proxy: torch.Tensor,
    n_segments: int = 3,
    temp: torch.Tensor = None,
    eps: float = 1e-6,
):
    b, h, c = r_proxy.shape
    if "_tgca_segment_slices_v7" in globals():
        slices = _tgca_segment_slices_v7(h, n_segments)
    else:
        slices = [(int(i * h / n_segments), int((i + 1) * h / n_segments)) for i in range(n_segments)]

    energies = []
    for st, ed in slices:
        e = r_proxy[:, st:ed, :].pow(2).mean(dim=(1, 2))
        energies.append(e)

    e = torch.stack(energies, dim=1)  # [B,S]

    e_min = e.min(dim=1, keepdim=True).values
    e_max = e.max(dim=1, keepdim=True).values
    h_energy = (e - e_min) / (e_max - e_min + eps)

    if temp is None:
        temp = torch.tensor(0.35, device=r_proxy.device, dtype=r_proxy.dtype)

    btm = torch.softmax(h_energy / temp.clamp_min(0.05), dim=1)
    return btm, h_energy, e


# Preserve a clean init base.
if not hasattr(TGCAForecaster, "_tgca_v8a_base_init"):
    if hasattr(TGCAForecaster, "_tgca_v7a_old_init"):
        TGCAForecaster._tgca_v8a_base_init = TGCAForecaster._tgca_v7a_old_init
    else:
        TGCAForecaster._tgca_v8a_base_init = TGCAForecaster.__init__


def _tgca_init_v8a_boltzmann_segment_prior(self, cfg: TGCAConfig):
    TGCAForecaster._tgca_v8a_base_init(self, cfg)

    # Segment-wise heads from V7A arch-only.
    self.seg_op_head = nn.Linear(cfg.d_state, cfg.n_segments * cfg.n_ops)
    self.seg_alpha_head = nn.Linear(cfg.d_state, cfg.n_segments)

    nn.init.zeros_(self.seg_op_head.weight)
    nn.init.zeros_(self.seg_op_head.bias)
    nn.init.zeros_(self.seg_alpha_head.weight)
    nn.init.zeros_(self.seg_alpha_head.bias)

    # Boltzmann hard-mode prior parameters.
    # gamma is deliberately small; starts around 0.04.
    self.btm_gamma_logit = nn.Parameter(torch.tensor(-0.7))
    self.btm_temp_logit = nn.Parameter(torch.tensor(-0.3))

    if "HardTrustProjectorV7A" in globals():
        self.projector = HardTrustProjectorV7A(cfg)


def _tgca_forward_v8a_boltzmann_segment_prior(
    self,
    x_enc: torch.Tensor,
    x_mark_enc: Optional[torch.Tensor] = None,
    x_dec: Optional[torch.Tensor] = None,
    x_mark_dec: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    y_true: Optional[torch.Tensor] = None,
    return_aux: bool = True,
):
    x_norm, mean, std = self._norm(x_enc)
    y0_norm = self.base(x_norm)

    y_true_norm = (y_true - mean) / std if y_true is not None else None

    state = self.state(x_norm, y0_norm)

    branch_probs_pred, op_probs_global, alpha_branch = self.router(
        state["branch_logits"],
        state["op_logits"],
    )

    # Detached diagnostic branch, same as v5/v7.
    if hasattr(self.state, "branch_head"):
        branch_logits_diag = self.state.branch_head(state["z"].detach())
        branch_probs_diag = torch.softmax(branch_logits_diag / self.cfg.router_tau, dim=-1)
    else:
        branch_probs_diag = branch_probs_pred.detach()

    deltas = self.bank(y0_norm, state["z"])
    b, _, h, c = deltas.shape
    s_num = self.cfg.n_segments

    if "_tgca_segment_slices_v7" in globals():
        slices = _tgca_segment_slices_v7(h, s_num)
    else:
        slices = [(int(i * h / s_num), int((i + 1) * h / s_num)) for i in range(s_num)]

    seg_op_logits = self.seg_op_head(state["z"]).view(b, s_num, self.cfg.n_ops)
    seg_op_probs = torch.softmax(seg_op_logits / self.cfg.router_tau, dim=-1)

    delta_raw = torch.zeros(b, h, c, device=x_enc.device, dtype=x_enc.dtype)
    for i, (st, ed) in enumerate(slices):
        w = seg_op_probs[:, i, :, None, None]
        delta_raw[:, st:ed, :] = (deltas[:, :, st:ed, :] * w).sum(dim=1)

    alpha_raw_seg = torch.sigmoid(self.seg_alpha_head(state["z"])).view(b, s_num, 1)

    # Boltzmann hard-mode segment prior from r_proxy.
    temp = 0.15 + 0.70 * torch.sigmoid(self.btm_temp_logit)
    gamma = 0.12 * torch.sigmoid(self.btm_gamma_logit)

    btm_seg, hard_energy_seg, raw_energy_seg = _tgca_segment_hard_boltzmann_v8(
        state["r_proxy"],
        n_segments=s_num,
        temp=temp,
        eps=self.cfg.eps,
    )

    # Uniform btm means no change.
    # Harder-than-uniform segments get mild boost; easier segments get mild damping.
    hard_scale_seg = 1.0 + gamma * (btm_seg * s_num - 1.0)
    hard_scale_seg = hard_scale_seg.clamp(0.85, 1.15)

    alpha_raw_seg = (alpha_raw_seg * hard_scale_seg.unsqueeze(-1)).clamp(0.0, 1.0)

    y_norm, trust_info = self.projector(
        y0=y0_norm,
        delta_raw=delta_raw,
        r_proxy=state["r_proxy"],
        alpha_raw=alpha_raw_seg,
        alpha_branch=alpha_branch,
        y_true_norm=y_true_norm,
    )

    if y_true_norm is not None:
        r_base_norm = y0_norm - y_true_norm

        if "_tgca_make_branch_oracle_v2" in globals():
            trust_info["branch_oracle"] = _tgca_make_branch_oracle_v2(
                r_base=r_base_norm,
                delta_raw=delta_raw,
                alpha_energy=trust_info["alpha_energy"],
                js_raw=trust_info["js_raw"],
                kl_raw=trust_info["kl_raw"],
                rkl_raw=trust_info["rkl_raw"],
                kl_asym_raw=trust_info["kl_asym_raw"],
                cfg=self.cfg,
            )
        elif "make_tgca_branch_oracle" in globals():
            trust_info["branch_oracle"] = make_tgca_branch_oracle(
                r_base=r_base_norm,
                delta_raw=delta_raw,
                alpha_energy=trust_info["alpha_energy"],
                js_raw=trust_info["js_raw"],
                kl_raw=trust_info["rkl_raw"],
                rkl_raw=trust_info["rkl_raw"],
                kl_asym_raw=trust_info["kl_asym_raw"],
                cfg=self.cfg,
            )

    y = self._denorm(y_norm, mean, std)
    y0 = self._denorm(y0_norm, mean, std)
    r_proxy = state["r_proxy"] * std

    if not return_aux:
        return y

    aux = {
        "y0": y0,
        "y_norm": y_norm,
        "y0_norm": y0_norm,
        "delta_raw_norm": delta_raw,
        "r_proxy": r_proxy,
        "r_proxy_norm": state["r_proxy"],
        "branch_probs": branch_probs_diag,
        "branch_probs_pred": branch_probs_pred,
        "op_probs": op_probs_global,
        "seg_op_probs": seg_op_probs,
        "deltas_norm": deltas,
        "mean": mean,
        "std": std,

        # V8A audit signals.
        "btm_seg": btm_seg,
        "hard_energy_seg": hard_energy_seg,
        "raw_energy_seg": raw_energy_seg,
        "hard_scale_seg": hard_scale_seg,
        "btm_gamma": gamma.detach().view(1),
        "btm_temp": temp.detach().view(1),
    }

    aux.update(trust_info)
    return y, aux


TGCAForecaster.__init__ = _tgca_init_v8a_boltzmann_segment_prior
TGCAForecaster.forward = _tgca_forward_v8a_boltzmann_segment_prior

# ===== TGCA_V8A_BOLTZMANN_SEGMENT_PRIOR_END =====




# ===== TGCA_V8B_DUAL_SAFE_SEGMENT_GATE_START =====
# V8B:
#   Mechanism directly from the uploaded documents:
#   MSE-safe + MAE-safe segment gate.
#
#   MSE condition:
#       sum(2<r0, delta> + ||delta||^2) < 0
#   MAE first-order condition:
#       sum(<sign(r0), delta>) < 0
#
#   We use r_proxy as deployable residual proxy, not y_true.
#   This keeps V7A segment-wise alpha/op routing and adds a mild pattern-safe gate.

def _tgca_dual_safe_segment_gate_v8b(
    r_proxy: torch.Tensor,
    delta_raw: torch.Tensor,
    n_segments: int = 3,
    eps: float = 1e-6,
):
    b, h, c = r_proxy.shape

    if "_tgca_segment_slices_v7" in globals():
        slices = _tgca_segment_slices_v7(h, n_segments)
    else:
        slices = [(int(i * h / n_segments), int((i + 1) * h / n_segments)) for i in range(n_segments)]

    mse_scores = []
    mae_scores = []
    gates = []

    for st, ed in slices:
        r = r_proxy[:, st:ed, :]
        d = delta_raw[:, st:ed, :]

        # Document MSE expansion term: 2<r, delta> + ||delta||^2.
        mse_raw = (2.0 * r * d + d.pow(2)).mean(dim=(1, 2))

        # Document MAE first-order term: <sign(r), delta>.
        mae_raw = (torch.sign(r) * d).mean(dim=(1, 2))

        # Normalize for stable sigmoid scale.
        mse_den = (2.0 * torch.abs(r * d) + d.pow(2)).mean(dim=(1, 2)).detach() + eps
        mae_den = torch.abs(d).mean(dim=(1, 2)).detach() + eps

        mse_norm = mse_raw / mse_den
        mae_norm = mae_raw / mae_den

        # Negative score is safe. Positive score is risky.
        mse_safe = torch.sigmoid(-4.0 * mse_norm)
        mae_safe = torch.sigmoid(-4.0 * mae_norm)

        safe = 0.5 * (mse_safe + mae_safe)

        # Mild structural gate:
        # safe≈0.5 -> gate≈1.0
        # highly unsafe -> 0.90
        # highly safe -> 1.10
        gate = 0.90 + 0.20 * safe

        mse_scores.append(mse_norm)
        mae_scores.append(mae_norm)
        gates.append(gate)

    mse_scores = torch.stack(mse_scores, dim=1)  # [B,S]
    mae_scores = torch.stack(mae_scores, dim=1)
    gates = torch.stack(gates, dim=1).clamp(0.90, 1.10)

    return gates, mse_scores, mae_scores


# Force back to the proven V7A arch-only init.
# This disables V8A Boltzmann alpha reweighting while keeping segment heads.
if "_tgca_init_v7a" in globals():
    TGCAForecaster.__init__ = _tgca_init_v7a

if "HardTrustProjectorV7A" in globals():
    HardTrustProjector = HardTrustProjectorV7A


def _tgca_forward_v8b_dual_safe_segment_gate(
    self,
    x_enc: torch.Tensor,
    x_mark_enc: Optional[torch.Tensor] = None,
    x_dec: Optional[torch.Tensor] = None,
    x_mark_dec: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    y_true: Optional[torch.Tensor] = None,
    return_aux: bool = True,
):
    # Start from current best V7A segment-wise predictor.
    if "_tgca_forward_v7a_segment_tail" not in globals():
        raise RuntimeError("V7A forward is required for V8B but not found")

    y_v7a, aux = _tgca_forward_v7a_segment_tail(
        self,
        x_enc=x_enc,
        x_mark_enc=x_mark_enc,
        x_dec=x_dec,
        x_mark_dec=x_mark_dec,
        mask=mask,
        y_true=y_true,
        return_aux=True,
    )

    y0 = aux["y0"]
    r_proxy_norm = aux.get("r_proxy_norm", None)
    delta_raw_norm = aux.get("delta_raw_norm", None)

    if r_proxy_norm is None or delta_raw_norm is None:
        if not return_aux:
            return y_v7a
        return y_v7a, aux

    b, h, c = delta_raw_norm.shape
    s_num = self.cfg.n_segments

    gates, mse_scores, mae_scores = _tgca_dual_safe_segment_gate_v8b(
        r_proxy=r_proxy_norm,
        delta_raw=delta_raw_norm,
        n_segments=s_num,
        eps=self.cfg.eps,
    )

    if "_tgca_segment_slices_v7" in globals():
        slices = _tgca_segment_slices_v7(h, s_num)
    else:
        slices = [(int(i * h / s_num), int((i + 1) * h / s_num)) for i in range(s_num)]

    gate_map = torch.ones(b, h, 1, device=y_v7a.device, dtype=y_v7a.dtype)
    for i, (st, ed) in enumerate(slices):
        gate_map[:, st:ed, :] = gates[:, i:i+1].view(b, 1, 1)

    # Apply gate in output space to V7A correction.
    y = y0 + gate_map * (y_v7a - y0)

    if "mean" in aux and "std" in aux:
        aux["y_norm"] = (y - aux["mean"]) / aux["std"]

    aux["dual_safe_gate_seg"] = gates
    aux["dual_safe_mse_score_seg"] = mse_scores
    aux["dual_safe_mae_score_seg"] = mae_scores

    if "alpha_seg" in aux and aux["alpha_seg"].ndim == 2 and aux["alpha_seg"].shape == gates.shape:
        aux["alpha_seg_before_dual_safe"] = aux["alpha_seg"]
        aux["alpha_seg"] = aux["alpha_seg"] * gates
        aux["alpha"] = aux["alpha_seg"].mean(dim=1, keepdim=True).unsqueeze(-1)

    if not return_aux:
        return y

    return y, aux


TGCAForecaster.forward = _tgca_forward_v8b_dual_safe_segment_gate

# ===== TGCA_V8B_DUAL_SAFE_SEGMENT_GATE_END =====




# ===== TGCA_V8C_SEGMENT_SPECTRAL_TRANSPORT_START =====
# V8C:
#   Mechanism from the two uploaded documents:
#   spectral transport / missing-mode coverage.
#
#   Key design:
#   - Keep the proven V7A arch-only segment alpha/operator.
#   - Do not gate alpha.
#   - Do not add spectral loss.
#   - Add a very mild segment-wise spectral transport operator on delta_raw.
#
#   For each segment:
#       delta_freq = rfft(delta_seg)
#       p_r = spectral probability of residual proxy
#       gain = 1 + strength * normalized(p_r - uniform)
#       delta_seg = irfft(gain * delta_freq)
#
#   This implements "move correction capacity toward residual-dominant spectral modes"
#   without using y_true and without hard safety gating.

def _tgca_apply_segment_spectral_transport_v8c(
    r_proxy: torch.Tensor,
    delta_raw: torch.Tensor,
    n_segments: int = 3,
    strength: torch.Tensor = None,
    eps: float = 1e-6,
):
    b, h, c = delta_raw.shape

    if "_tgca_segment_slices_v7" in globals():
        slices = _tgca_segment_slices_v7(h, n_segments)
    else:
        slices = [(int(i * h / n_segments), int((i + 1) * h / n_segments)) for i in range(n_segments)]

    if strength is None:
        strength = torch.tensor(0.02, device=delta_raw.device, dtype=delta_raw.dtype)

    out = torch.zeros_like(delta_raw)
    w1_list = []
    gain_dev_list = []
    tail_mass_list = []

    for st, ed in slices:
        r = r_proxy[:, st:ed, :]
        d = delta_raw[:, st:ed, :]
        seg_len = ed - st

        rf = torch.fft.rfft(r, dim=1)
        df = torch.fft.rfft(d, dim=1)

        er = rf.abs().pow(2).mean(dim=2) + eps      # [B,F]
        edel = df.abs().pow(2).mean(dim=2) + eps    # [B,F]

        pr = er / er.sum(dim=1, keepdim=True).clamp_min(eps)
        pd = edel / edel.sum(dim=1, keepdim=True).clamp_min(eps)

        f = pr.size(1)
        uniform = torch.full_like(pr, 1.0 / max(f, 1))

        centered = pr - uniform
        centered = centered / centered.abs().amax(dim=1, keepdim=True).clamp_min(eps)

        gain = (1.0 + strength * centered).clamp(0.95, 1.05)  # conservative
        gain_c = gain.unsqueeze(-1)

        dt = torch.fft.irfft(df * gain_c, n=seg_len, dim=1)
        out[:, st:ed, :] = dt

        # W1 proxy by CDF difference, for audit.
        cdf_r = torch.cumsum(pr, dim=1)
        cdf_d = torch.cumsum(pd, dim=1)
        w1 = (cdf_r - cdf_d).abs().mean(dim=1)
        w1_list.append(w1)

        gain_dev_list.append((gain - 1.0).abs().mean(dim=1))

        # high-frequency half mass as a cheap tail spectral audit
        half = f // 2
        tail_mass_list.append(pr[:, half:].sum(dim=1))

    w1_proxy = torch.stack(w1_list, dim=1)          # [B,S]
    gain_dev = torch.stack(gain_dev_list, dim=1)    # [B,S]
    tail_mass = torch.stack(tail_mass_list, dim=1)  # [B,S]

    return out, {
        "spectral_transport_w1_seg": w1_proxy,
        "spectral_transport_gain_dev_seg": gain_dev,
        "spectral_tail_mass_seg": tail_mass,
    }


# Force clean V7A arch-only init, avoiding V8A/V8B stacked gates.
if not hasattr(TGCAForecaster, "_tgca_v8c_clean_base_init"):
    if hasattr(TGCAForecaster, "_tgca_v7a_old_init"):
        TGCAForecaster._tgca_v8c_clean_base_init = TGCAForecaster._tgca_v7a_old_init
    else:
        TGCAForecaster._tgca_v8c_clean_base_init = TGCAForecaster.__init__


def _tgca_init_v8c_segment_spectral_transport(self, cfg: TGCAConfig):
    TGCAForecaster._tgca_v8c_clean_base_init(self, cfg)

    # V7A arch-only segment heads.
    self.seg_op_head = nn.Linear(cfg.d_state, cfg.n_segments * cfg.n_ops)
    self.seg_alpha_head = nn.Linear(cfg.d_state, cfg.n_segments)

    nn.init.zeros_(self.seg_op_head.weight)
    nn.init.zeros_(self.seg_op_head.bias)
    nn.init.zeros_(self.seg_alpha_head.weight)
    nn.init.zeros_(self.seg_alpha_head.bias)

    # Tiny trainable spectral transport strength.
    # strength = 0.06 * sigmoid(logit), init roughly 0.011
    self.spec_transport_logit = nn.Parameter(torch.tensor(-1.5))

    if "HardTrustProjectorV7A" in globals():
        self.projector = HardTrustProjectorV7A(cfg)


def _tgca_forward_v8c_segment_spectral_transport(
    self,
    x_enc: torch.Tensor,
    x_mark_enc: Optional[torch.Tensor] = None,
    x_dec: Optional[torch.Tensor] = None,
    x_mark_dec: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    y_true: Optional[torch.Tensor] = None,
    return_aux: bool = True,
):
    x_norm, mean, std = self._norm(x_enc)
    y0_norm = self.base(x_norm)
    y_true_norm = (y_true - mean) / std if y_true is not None else None

    state = self.state(x_norm, y0_norm)

    branch_probs_pred, op_probs_global, alpha_branch = self.router(
        state["branch_logits"],
        state["op_logits"],
    )

    # Detached diagnostic branch, inherited from the best v5/v7 idea.
    if hasattr(self.state, "branch_head"):
        branch_logits_diag = self.state.branch_head(state["z"].detach())
        branch_probs_diag = torch.softmax(branch_logits_diag / self.cfg.router_tau, dim=-1)
    else:
        branch_probs_diag = branch_probs_pred.detach()

    deltas = self.bank(y0_norm, state["z"])  # [B,n_ops,H,C]
    b, _, h, c = deltas.shape
    s_num = self.cfg.n_segments

    if "_tgca_segment_slices_v7" in globals():
        slices = _tgca_segment_slices_v7(h, s_num)
    else:
        slices = [(int(i * h / s_num), int((i + 1) * h / s_num)) for i in range(s_num)]

    seg_op_logits = self.seg_op_head(state["z"]).view(b, s_num, self.cfg.n_ops)
    seg_op_probs = torch.softmax(seg_op_logits / self.cfg.router_tau, dim=-1)

    delta_raw = torch.zeros(b, h, c, device=x_enc.device, dtype=x_enc.dtype)
    for i, (st, ed) in enumerate(slices):
        w = seg_op_probs[:, i, :, None, None]
        delta_raw[:, st:ed, :] = (deltas[:, :, st:ed, :] * w).sum(dim=1)

    # V8C spectral transport, conservative and segment-wise.
    spec_strength = 0.06 * torch.sigmoid(self.spec_transport_logit)
    delta_raw, spec_aux = _tgca_apply_segment_spectral_transport_v8c(
        r_proxy=state["r_proxy"],
        delta_raw=delta_raw,
        n_segments=s_num,
        strength=spec_strength,
        eps=self.cfg.eps,
    )

    alpha_raw_seg = torch.sigmoid(self.seg_alpha_head(state["z"])).view(b, s_num, 1)

    y_norm, trust_info = self.projector(
        y0=y0_norm,
        delta_raw=delta_raw,
        r_proxy=state["r_proxy"],
        alpha_raw=alpha_raw_seg,
        alpha_branch=alpha_branch,
        y_true_norm=y_true_norm,
    )

    if y_true_norm is not None:
        r_base_norm = y0_norm - y_true_norm

        if "_tgca_make_branch_oracle_v2" in globals():
            trust_info["branch_oracle"] = _tgca_make_branch_oracle_v2(
                r_base=r_base_norm,
                delta_raw=delta_raw,
                alpha_energy=trust_info["alpha_energy"],
                js_raw=trust_info["js_raw"],
                kl_raw=trust_info["kl_raw"],
                rkl_raw=trust_info["rkl_raw"],
                kl_asym_raw=trust_info["kl_asym_raw"],
                cfg=self.cfg,
            )
        elif "make_tgca_branch_oracle" in globals():
            trust_info["branch_oracle"] = make_tgca_branch_oracle(
                r_base=r_base_norm,
                delta_raw=delta_raw,
                alpha_energy=trust_info["alpha_energy"],
                js_raw=trust_info["js_raw"],
                kl_raw=trust_info["kl_raw"],
                rkl_raw=trust_info["rkl_raw"],
                kl_asym_raw=trust_info["kl_asym_raw"],
                cfg=self.cfg,
            )

    y = self._denorm(y_norm, mean, std)
    y0 = self._denorm(y0_norm, mean, std)
    r_proxy = state["r_proxy"] * std

    if not return_aux:
        return y

    aux = {
        "y0": y0,
        "y_norm": y_norm,
        "y0_norm": y0_norm,
        "delta_raw_norm": delta_raw,
        "r_proxy": r_proxy,
        "r_proxy_norm": state["r_proxy"],
        "branch_probs": branch_probs_diag,
        "branch_probs_pred": branch_probs_pred,
        "op_probs": op_probs_global,
        "seg_op_probs": seg_op_probs,
        "deltas_norm": deltas,
        "mean": mean,
        "std": std,
        "spec_transport_strength": spec_strength.detach().view(1),
    }

    aux.update(spec_aux)
    aux.update(trust_info)
    return y, aux


TGCAForecaster.__init__ = _tgca_init_v8c_segment_spectral_transport
TGCAForecaster.forward = _tgca_forward_v8c_segment_spectral_transport

# ===== TGCA_V8C_SEGMENT_SPECTRAL_TRANSPORT_END =====




# ===== TGCA_V9_SEGMENT_PROTO_MEMORY_START =====
# V9:
#   Segment Prototype Recursive Memory.
#
#   Mechanism from uploaded documents:
#   - repeated samples should not all be damped;
#   - truth-positive / understep categories should be enhanced;
#   - wrong-direction / overstep categories should be damped;
#   - TimeBridge / long horizon should use segment-wise memory.
#
#   This version:
#   - starts from proven V7A arch-only;
#   - no extra loss;
#   - no hard gate;
#   - adds a small trainable prototype memory prior to segment op logits and segment alpha logits.

if not hasattr(TGCAForecaster, "_tgca_v9_clean_base_init"):
    if hasattr(TGCAForecaster, "_tgca_v7a_old_init"):
        TGCAForecaster._tgca_v9_clean_base_init = TGCAForecaster._tgca_v7a_old_init
    else:
        TGCAForecaster._tgca_v9_clean_base_init = TGCAForecaster.__init__


def _tgca_init_v9_segment_proto_memory(self, cfg: TGCAConfig):
    TGCAForecaster._tgca_v9_clean_base_init(self, cfg)

    # V7A arch-only heads.
    self.seg_op_head = nn.Linear(cfg.d_state, cfg.n_segments * cfg.n_ops)
    self.seg_alpha_head = nn.Linear(cfg.d_state, cfg.n_segments)

    nn.init.zeros_(self.seg_op_head.weight)
    nn.init.zeros_(self.seg_op_head.bias)
    nn.init.zeros_(self.seg_alpha_head.weight)
    nn.init.zeros_(self.seg_alpha_head.bias)

    # Prototype memory bank.
    k = 8
    self.proto_keys = nn.Parameter(torch.randn(k, cfg.d_state) * 0.02)

    # Memory priors are neutral at init.
    self.proto_alpha_bias = nn.Parameter(torch.zeros(k, cfg.n_segments))
    self.proto_op_bias = nn.Parameter(torch.zeros(k, cfg.n_segments, cfg.n_ops))

    # Small learnable strength. gamma ≈ 0.018 at init, max ≈ 0.10.
    self.proto_gamma_logit = nn.Parameter(torch.tensor(-1.5))

    if "HardTrustProjectorV7A" in globals():
        self.projector = HardTrustProjectorV7A(cfg)


def _tgca_proto_memory_prior_v9(self, z: torch.Tensor):
    eps = self.cfg.eps if hasattr(self, "cfg") else 1e-6

    z_norm = F.normalize(z, dim=-1, eps=eps)
    k_norm = F.normalize(self.proto_keys, dim=-1, eps=eps)

    dist = torch.cdist(z_norm.unsqueeze(1), k_norm.unsqueeze(0), p=2).squeeze(1)  # [B,K]

    # Soft nearest prototype.
    temp = 0.35
    proto_w = torch.softmax(-dist / temp, dim=-1)  # [B,K]

    alpha_prior = torch.einsum("bk,ks->bs", proto_w, self.proto_alpha_bias)
    op_prior = torch.einsum("bk,kso->bso", proto_w, self.proto_op_bias)

    gamma = 0.10 * torch.sigmoid(self.proto_gamma_logit)

    return alpha_prior, op_prior, proto_w, gamma


def _tgca_forward_v9_segment_proto_memory(
    self,
    x_enc: torch.Tensor,
    x_mark_enc: Optional[torch.Tensor] = None,
    x_dec: Optional[torch.Tensor] = None,
    x_mark_dec: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    y_true: Optional[torch.Tensor] = None,
    return_aux: bool = True,
):
    x_norm, mean, std = self._norm(x_enc)
    y0_norm = self.base(x_norm)

    y_true_norm = (y_true - mean) / std if y_true is not None else None

    state = self.state(x_norm, y0_norm)

    branch_probs_pred, op_probs_global, alpha_branch = self.router(
        state["branch_logits"],
        state["op_logits"],
    )

    # Detached diagnostic branch.
    if hasattr(self.state, "branch_head"):
        branch_logits_diag = self.state.branch_head(state["z"].detach())
        branch_probs_diag = torch.softmax(branch_logits_diag / self.cfg.router_tau, dim=-1)
    else:
        branch_probs_diag = branch_probs_pred.detach()

    deltas = self.bank(y0_norm, state["z"])
    b, _, h, c = deltas.shape
    s_num = self.cfg.n_segments

    if "_tgca_segment_slices_v7" in globals():
        slices = _tgca_segment_slices_v7(h, s_num)
    else:
        slices = [(int(i * h / s_num), int((i + 1) * h / s_num)) for i in range(s_num)]

    alpha_prior, op_prior, proto_w, proto_gamma = _tgca_proto_memory_prior_v9(self, state["z"])

    seg_op_logits = self.seg_op_head(state["z"]).view(b, s_num, self.cfg.n_ops)
    seg_op_logits = seg_op_logits + proto_gamma * op_prior
    seg_op_probs = torch.softmax(seg_op_logits / self.cfg.router_tau, dim=-1)

    delta_raw = torch.zeros(b, h, c, device=x_enc.device, dtype=x_enc.dtype)
    for i, (st, ed) in enumerate(slices):
        w = seg_op_probs[:, i, :, None, None]
        delta_raw[:, st:ed, :] = (deltas[:, :, st:ed, :] * w).sum(dim=1)

    alpha_logits = self.seg_alpha_head(state["z"]).view(b, s_num)
    alpha_logits = alpha_logits + proto_gamma * alpha_prior
    alpha_raw_seg = torch.sigmoid(alpha_logits).view(b, s_num, 1)

    y_norm, trust_info = self.projector(
        y0=y0_norm,
        delta_raw=delta_raw,
        r_proxy=state["r_proxy"],
        alpha_raw=alpha_raw_seg,
        alpha_branch=alpha_branch,
        y_true_norm=y_true_norm,
    )

    if y_true_norm is not None:
        r_base_norm = y0_norm - y_true_norm

        if "_tgca_make_branch_oracle_v2" in globals():
            trust_info["branch_oracle"] = _tgca_make_branch_oracle_v2(
                r_base=r_base_norm,
                delta_raw=delta_raw,
                alpha_energy=trust_info["alpha_energy"],
                js_raw=trust_info["js_raw"],
                kl_raw=trust_info["kl_raw"],
                rkl_raw=trust_info["rkl_raw"],
                kl_asym_raw=trust_info["kl_asym_raw"],
                cfg=self.cfg,
            )
        elif "make_tgca_branch_oracle" in globals():
            trust_info["branch_oracle"] = make_tgca_branch_oracle(
                r_base=r_base_norm,
                delta_raw=delta_raw,
                alpha_energy=trust_info["alpha_energy"],
                js_raw=trust_info["js_raw"],
                kl_raw=trust_info["kl_raw"],
                rkl_raw=trust_info["rkl_raw"],
                kl_asym_raw=trust_info["kl_asym_raw"],
                cfg=self.cfg,
            )

    y = self._denorm(y_norm, mean, std)
    y0 = self._denorm(y0_norm, mean, std)
    r_proxy = state["r_proxy"] * std

    if not return_aux:
        return y

    proto_entropy = -(proto_w.clamp_min(self.cfg.eps) * proto_w.clamp_min(self.cfg.eps).log()).sum(dim=-1)

    aux = {
        "y0": y0,
        "y_norm": y_norm,
        "y0_norm": y0_norm,
        "delta_raw_norm": delta_raw,
        "r_proxy": r_proxy,
        "r_proxy_norm": state["r_proxy"],
        "branch_probs": branch_probs_diag,
        "branch_probs_pred": branch_probs_pred,
        "op_probs": op_probs_global,
        "seg_op_probs": seg_op_probs,
        "deltas_norm": deltas,
        "mean": mean,
        "std": std,

        # V9 audit signals.
        "proto_w": proto_w,
        "proto_gamma": proto_gamma.detach().view(1),
        "proto_entropy": proto_entropy,
        "proto_alpha_prior": alpha_prior,
        "proto_op_prior_norm": op_prior.pow(2).mean(dim=(1,2)),
    }

    aux.update(trust_info)
    return y, aux


TGCAForecaster.__init__ = _tgca_init_v9_segment_proto_memory
TGCAForecaster.forward = _tgca_forward_v9_segment_proto_memory

# ===== TGCA_V9_SEGMENT_PROTO_MEMORY_END =====


if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = TGCAConfig(
        seq_len=96,
        pred_len=720,
        enc_in=7,
        d_state=128,
        moving_avg=25,
        use_revin=True,
    )

    model = TGCAForecaster(cfg)

    x = torch.randn(4, cfg.seq_len, cfg.enc_in)
    y = torch.randn(4, cfg.pred_len, cfg.enc_in)

    pred, aux = model(x, y_true=y, return_aux=True)
    loss, logs = tgca_forecasting_loss(pred, y, aux)

    print("TGCA_Forecaster smoke test OK")
    print("pred shape:", tuple(pred.shape))
    print("loss:", float(loss.detach().cpu()))
    for k, v in logs.items():
        print(f"{k}: {v}")
