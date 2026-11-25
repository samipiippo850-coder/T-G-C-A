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

from v9_delta_transplant_on_v0 import extract_y0_from_aux, forward_model as _pmrt_forward_model


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
    print("[PMRT-Hybrid] load:", ckpt_path)
    print("[PMRT-Hybrid] missing:", len(missing), "unexpected:", len(unexpected))
    if missing:
        print("[PMRT-Hybrid] missing first 10:", missing[:10])
    if unexpected:
        print("[PMRT-Hybrid] unexpected first 10:", unexpected[:10])


def call_model(model, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
    try:
        out = model(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            y_true=None,
            return_aux=True,
        )
    except TypeError:
        try:
            out = model(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                return_aux=True,
            )
        except TypeError:
            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    if isinstance(out, tuple):
        pred = out[0]
        aux = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
    else:
        pred, aux = out, {}

    return pred, aux


class Model(nn.Module):
    """
    PMRT-Hybrid:
        y_hat = y_hat_v0 + rho_raw * delta_raw + rho_mem * delta_memory

    where:
        delta_memory = y_hat_v9 - y0_v9
        delta_raw    = delta_raw_norm * std

    Environment variables:
        PMRT_CKPT0: v0 fast-path checkpoint
        PMRT_CKPT9: V9 prototype-memory checkpoint
        PMRT_RHO_RAW: default 0.6
        PMRT_RHO_MEM: default 0.0
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len

        self.rho_raw = float(os.environ.get("PMRT_RHO_RAW", "0.6"))
        self.rho_mem = float(os.environ.get("PMRT_RHO_MEM", "0.0"))

        ckpt0 = os.environ.get("PMRT_CKPT0", "")
        ckpt9 = os.environ.get("PMRT_CKPT9", "")
        if not ckpt0 or not ckpt9:
            raise RuntimeError("PMRT_CKPT0 and PMRT_CKPT9 must be set.")

        v0_src = os.environ.get(
            "PMRT_V0_MODEL",
            "models/TGCA_Forecaster.py.bak_lowmem_20260620_224011",
        )
        v9_src = os.environ.get(
            "PMRT_V9_MODEL",
            "models/TGCA_Forecaster.py.bak_v10a_hrmp_20260621_063859",
        )

        mod0 = import_model_file(v0_src, "pmrt_hybrid_v0_file")
        mod9 = import_model_file(v9_src, "pmrt_hybrid_v9_file")

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

        print("[PMRT-Hybrid] rho_raw =", self.rho_raw)
        print("[PMRT-Hybrid] rho_mem =", self.rho_mem)

    def _extract_delta_raw(self, aux9, pred_len, true_shape):
        if "delta_raw_norm" not in aux9:
            raise RuntimeError(f"aux9 has no delta_raw_norm. keys={sorted(aux9.keys())}")

        delta_raw = aux9["delta_raw_norm"]

        if "std" in aux9:
            delta_raw = delta_raw * aux9["std"]

        delta_raw = delta_raw[:, -pred_len:, :]
        if delta_raw.shape[-1] != true_shape[-1]:
            delta_raw = delta_raw[..., -true_shape[-1]:]

        return delta_raw

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
        y_true = None

        with torch.no_grad():
            pred0, aux0 = _pmrt_forward_model(self.v0, x_enc, x_mark_enc, x_mark_dec)
            pred9, aux9 = _pmrt_forward_model(self.v9, x_enc, x_mark_enc, x_mark_dec)

            pred0 = pred0[:, -self.pred_len:, :]
            pred9 = pred9[:, -self.pred_len:, :]

            if pred9.shape[-1] != pred0.shape[-1]:
                pred9 = pred9[..., -pred0.shape[-1]:]

            delta_raw = self._extract_delta_raw(aux9, self.pred_len, pred0.shape)

            # Raw-only mode does not need y0_v9. This avoids unnecessary failure
            # when rho_mem=0.0 and only delta_raw_norm is used.
            need_memory = abs(float(self.rho_mem)) > 1e-12 or return_aux

            y0_v9 = None
            y0_source = "not_used"
            delta_memory = torch.zeros_like(pred0)

            if need_memory:
                y0_v9, y0_source = extract_y0_from_aux(
                    aux9,
                    pred9,
                    self.pred_len,
                    pred0.shape,
                )

                if y0_v9.shape[-1] != pred0.shape[-1]:
                    y0_v9 = y0_v9[..., -pred0.shape[-1]:]

                delta_memory = pred9 - y0_v9

            pred = pred0 + self.rho_raw * delta_raw + self.rho_mem * delta_memory

            if return_aux:
                aux = {
                    "pred0": pred0,
                    "pred9": pred9,
                    "y0_v9": y0_v9 if y0_v9 is not None else torch.zeros_like(pred0),
                    "delta_raw": delta_raw,
                    "delta_memory": delta_memory,
                    "rho_raw": pred.new_tensor(self.rho_raw),
                    "rho_mem": pred.new_tensor(self.rho_mem),
                    "y0_source": y0_source,
                    "aux9_keys": sorted(list(aux9.keys())) if isinstance(aux9, dict) else [],
                }
                return pred, aux

            return pred
