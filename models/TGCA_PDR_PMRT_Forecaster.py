import os
import sys
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path("/root/DistDF-main/DistDF-main_competition").resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/tgca_fastpath_migration_v1"))

from v9_delta_transplant_on_v0 import (
    forward_model as _pmrt_forward_model,
    extract_y0_from_aux,
)


def import_model_file(path: str, module_name: str):
    path = str(Path(path).resolve())
    loader = SourceFileLoader(module_name, path)
    spec = importlib.util.spec_from_loader(module_name, loader, origin=path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model file: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def unwrap_state(raw):
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"]
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def load_ckpt(model, ckpt_path):
    raw = torch.load(ckpt_path, map_location="cpu")
    state = unwrap_state(raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[PDR-PMRT] load:", ckpt_path)
    print("[PDR-PMRT] missing:", len(missing), "unexpected:", len(unexpected))
    if missing:
        print("[PDR-PMRT] missing first 10:", missing[:10])
    if unexpected:
        print("[PDR-PMRT] unexpected first 10:", unexpected[:10])


def spectral_tilt(delta, gamma: float):
    """
    delta_gamma[f] = delta[f] * f^{-gamma}

    Power spectrum approximately becomes:
        P_gamma(f) = P(f) * f^{-2gamma}
    so beta is approximately increased by 2 * gamma.
    """
    if abs(float(gamma)) < 1e-12:
        return delta

    B, T, C = delta.shape
    X = torch.fft.rfft(delta, dim=1)
    Freq = X.shape[1]

    f = torch.arange(Freq, device=delta.device, dtype=delta.dtype).clamp_min(1.0)
    w = f.pow(-float(gamma))
    w[0] = 1.0
    w = w.view(1, Freq, 1)

    out = torch.fft.irfft(X * w, n=T, dim=1)
    return out.to(delta.dtype)


def slice_like(t, pred_len, ref_shape):
    t = t[:, -pred_len:, :]
    if t.shape[-1] != ref_shape[-1]:
        t = t[..., -ref_shape[-1]:]
    return t


def get_aux_delta(aux, key, pred_len, ref_shape):
    if key not in aux or not isinstance(aux[key], torch.Tensor):
        return None

    t = aux[key]

    # op-bank style: [B, K, T, C] -> mean over K
    if t.dim() == 4:
        t = t.mean(dim=1)

    # normalized residuals need to be mapped back to original scale
    if key.endswith("_norm") and "std" in aux:
        t = t * aux["std"]

    return slice_like(t, pred_len, ref_shape)


class Model(nn.Module):
    """
    PDR-PMRT: Power-law Direction-Repaired PMRT

        y_hat = y0
              + Tilt_{gamma_pos}(rho_raw * delta_raw)
              + eta * Tilt_{gamma_flip}(-deltas_norm_mean)

    Default score-best config from 4-seed sweep:
        rho_raw   = 0.6
        gamma_pos = 0.15
        eta       = 0.3
        gamma_flip= 0.0

    Environment variables:
        PDR_CKPT0 / PMRT_CKPT0
        PDR_CKPT9 / PMRT_CKPT9
        PDR_RHO_RAW
        PDR_GAMMA_POS
        PDR_ETA
        PDR_GAMMA_FLIP
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len

        self.rho_raw = float(os.environ.get("PDR_RHO_RAW", "0.6"))
        self.gamma_pos = float(os.environ.get("PDR_GAMMA_POS", "0.15"))
        self.eta = float(os.environ.get("PDR_ETA", "0.3"))
        self.gamma_flip = float(os.environ.get("PDR_GAMMA_FLIP", "0.0"))

        ckpt0 = os.environ.get("PDR_CKPT0", os.environ.get("PMRT_CKPT0", ""))
        ckpt9 = os.environ.get("PDR_CKPT9", os.environ.get("PMRT_CKPT9", ""))

        if not ckpt0 or not ckpt9:
            raise RuntimeError("PDR_CKPT0/PDR_CKPT9 or PMRT_CKPT0/PMRT_CKPT9 must be set.")

        v0_src = os.environ.get(
            "PDR_V0_MODEL",
            "models/TGCA_Forecaster.py.bak_lowmem_20260620_224011",
        )
        v9_src = os.environ.get(
            "PDR_V9_MODEL",
            "models/TGCA_Forecaster.py.bak_v10a_hrmp_20260621_063859",
        )

        mod0 = import_model_file(v0_src, "pdr_pmrt_v0_file")
        mod9 = import_model_file(v9_src, "pdr_pmrt_v9_file")

        self.v0 = mod0.Model(args).float()
        self.v9 = mod9.Model(args).float()

        load_ckpt(self.v0, ckpt0)
        load_ckpt(self.v9, ckpt9)

        self.v0.eval()
        self.v9.eval()

        for p in self.v0.parameters():
            p.requires_grad_(False)
        for p in self.v9.parameters():
            p.requires_grad_(False)

        print("[PDR-PMRT] rho_raw    =", self.rho_raw)
        print("[PDR-PMRT] gamma_pos  =", self.gamma_pos)
        print("[PDR-PMRT] eta        =", self.eta)
        print("[PDR-PMRT] gamma_flip =", self.gamma_flip)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
        y_true=None,
        return_aux=False,
    ):
        # hard no-ytrue guard
        y_true = None

        with torch.no_grad():
            pred0, aux0 = _pmrt_forward_model(self.v0, x_enc, x_mark_enc, x_mark_dec)
            pred9, aux9 = _pmrt_forward_model(self.v9, x_enc, x_mark_enc, x_mark_dec)

            pred0 = pred0[:, -self.pred_len:, :]
            pred9 = pred9[:, -self.pred_len:, :]

            if pred9.shape[-1] != pred0.shape[-1]:
                pred9 = pred9[..., -pred0.shape[-1]:]

            delta_raw = get_aux_delta(aux9, "delta_raw_norm", self.pred_len, pred0.shape)
            if delta_raw is None:
                raise RuntimeError("PDR-PMRT requires aux9['delta_raw_norm'].")

            deltas_norm = get_aux_delta(aux9, "deltas_norm", self.pred_len, pred0.shape)
            if deltas_norm is None:
                raise RuntimeError("PDR-PMRT requires aux9['deltas_norm'].")

            raw_branch = spectral_tilt(self.rho_raw * delta_raw, self.gamma_pos)

            # direction-repaired negative branch
            flip_branch = spectral_tilt(-deltas_norm, self.gamma_flip)

            pred = pred0 + raw_branch + self.eta * flip_branch

            if return_aux:
                aux = {
                    "pred0": pred0,
                    "pred9": pred9,
                    "delta_raw": delta_raw,
                    "deltas_norm": deltas_norm,
                    "raw_branch": raw_branch,
                    "flip_branch": flip_branch,
                    "rho_raw": pred.new_tensor(self.rho_raw),
                    "gamma_pos": pred.new_tensor(self.gamma_pos),
                    "eta": pred.new_tensor(self.eta),
                    "gamma_flip": pred.new_tensor(self.gamma_flip),
                    "aux9_keys": sorted(list(aux9.keys())) if isinstance(aux9, dict) else [],
                }
                return pred, aux

            return pred
