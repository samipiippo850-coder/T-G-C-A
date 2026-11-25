import os
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _env_on(name, default="0"):
    return str(os.environ.get(name, default)).lower() in ["1", "true", "yes", "y", "on"]


ACTIONS = ["base", "amp_0.85", "amp_0.95", "amp_1.05", "amp_1.15", "smooth5"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}


def _map_route_action(a):
    a = str(a)
    if a in ACTION_TO_ID:
        return a
    if a in ["calib_bias", "head_bias", "bias"]:
        return "base"
    return "base"


class TGCAMountedAdapter(nn.Module):
    """
    TGCA mounted v1.

    Route-init version:
    - action mixture is initialized from old TGCA selected_action
    - learned_bias is injected as per-channel bias path
    - no true label is used in forward
    """
    def __init__(
        self,
        pred_len,
        c_out,
        alpha_init=-8.0,
        beta_init=-8.0,
        route_csv=None,
        dataset=None,
        backbone=None,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.c_out = int(c_out)
        self.n_actions = len(ACTIONS)

        other_logit = float(os.environ.get("TGCA_ACTION_OTHER_LOGIT", "-1.0"))
        base_logit = float(os.environ.get("TGCA_ACTION_BASE_LOGIT", "2.0"))
        logits = torch.full((self.c_out, self.n_actions), other_logit)
        logits[:, ACTION_TO_ID["base"]] = base_logit

        channel_bias = torch.zeros(1, 1, self.c_out)
        alpha_logit = torch.full((1, 1, self.c_out), float(alpha_init))
        beta_logit = torch.full((1, 1, self.c_out), float(beta_init))

        loaded = self._load_route_init(
            logits=logits,
            channel_bias=channel_bias,
            alpha_logit=alpha_logit,
            beta_logit=beta_logit,
            route_csv=route_csv,
            dataset=dataset,
            backbone=backbone,
        )

        self.action_logits = nn.Parameter(logits)
        self.channel_bias = nn.Parameter(channel_bias)
        self.alpha_logit = nn.Parameter(alpha_logit)
        self.beta_logit = nn.Parameter(beta_logit)

        kernel = torch.ones(self.c_out, 1, 5) / 5.0
        self.register_buffer("smooth_kernel", kernel)

        self.route_init_loaded = loaded

    def _load_route_init(self, logits, channel_bias, alpha_logit, beta_logit, route_csv, dataset, backbone):
        if not _env_on("TGCA_ROUTE_INIT_ENABLE", "0"):
            return 0

        if not route_csv:
            return 0

        p = Path(route_csv)
        if not p.exists():
            print(f"[TGCA_ROUTE_INIT_WARN] route csv not found: {p}", flush=True)
            return 0

        loaded = 0
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("dataset", "")) != str(dataset):
                    continue
                if str(row.get("backbone", "")) != str(backbone):
                    continue

                ch = int(row["channel"])
                if ch < 0 or ch >= self.c_out:
                    continue

                selected_raw = row.get("selected_action", "base")
                selected = _map_route_action(selected_raw)
                aid = ACTION_TO_ID[selected]

                logits[ch, :] = -4.0
                logits[ch, aid] = 4.0

                # 如果是幅度/平滑动作，就强打开 action path；如果只是 calib_bias，则 action path 保持接近 base。
                if selected != "base":
                    alpha_logit[0, 0, ch] = 2.0
                else:
                    alpha_logit[0, 0, ch] = -8.0

                try:
                    b = float(row.get("learned_bias", 0.0))
                except Exception:
                    b = 0.0

                channel_bias[0, 0, ch] = b

                # old router 的 learned_bias 基本是通道偏置校准，v1 直接打开 bias path。
                if abs(b) > 1e-8:
                    beta_logit[0, 0, ch] = 2.0
                else:
                    beta_logit[0, 0, ch] = -8.0

                loaded += 1

        print(
            f"[TGCA_ROUTE_INIT] dataset={dataset} backbone={backbone} loaded_channels={loaded} csv={route_csv}",
            flush=True,
        )
        return loaded

    def _smooth5(self, y):
        x = y.transpose(1, 2)
        x = F.pad(x, (2, 2), mode="replicate")
        z = F.conv1d(x, self.smooth_kernel, groups=self.c_out)
        return z.transpose(1, 2)

    def _amp(self, y, scale):
        mu = y.mean(dim=1, keepdim=True)
        return mu + float(scale) * (y - mu)

    def forward(self, y):
        if y is None or (not torch.is_tensor(y)):
            return y
        if y.ndim != 3:
            return y
        if y.shape[-1] != self.c_out:
            return y

        cand = torch.stack([
            y,
            self._amp(y, 0.85),
            self._amp(y, 0.95),
            self._amp(y, 1.05),
            self._amp(y, 1.15),
            self._smooth5(y),
        ], dim=-1)

        w = torch.softmax(self.action_logits, dim=-1).view(1, 1, self.c_out, self.n_actions)
        mix = (cand * w).sum(dim=-1)

        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)

        # Ablation switches. No true labels are used.
        if _env_on("TGCA_DISABLE_ACTION", "0"):
            alpha = torch.zeros_like(alpha)
        if _env_on("TGCA_DISABLE_BIAS", "0"):
            beta = torch.zeros_like(beta)

        out = y + alpha * (mix - y) + beta * self.channel_bias

        # TGCA mounted v2 safety gate:
        # pred-only trust projection. No true labels are used.
        if _env_on("TGCA_SAFETY_GATE", "0"):
            cap = float(os.environ.get("TGCA_SAFETY_CAP", "0.035"))
            delta = out - y
            base_norm = torch.sqrt(torch.mean(y * y, dim=(1, 2), keepdim=True) + 1e-8)
            delta_norm = torch.sqrt(torch.mean(delta * delta, dim=(1, 2), keepdim=True) + 1e-8)
            ratio = delta_norm / (base_norm + 1e-8)
            scale = torch.clamp(cap / (ratio + 1e-8), max=1.0)
            out = y + scale * delta

        if _env_on("TGCA_FORWARD_AUDIT", "0") and not hasattr(self, "_tgca_forward_audit_printed"):
            with torch.no_grad():
                d = out - y
                base_norm0 = torch.sqrt(torch.mean(y * y) + 1e-8).item()
                delta_norm0 = torch.sqrt(torch.mean(d * d) + 1e-8).item()
                delta_ratio0 = delta_norm0 / (base_norm0 + 1e-8)
                max_abs_delta0 = torch.max(torch.abs(d)).item()
                print(
                    f"[TGCA_FORWARD_AUDIT] base_norm={base_norm0:.8g} "
                    f"delta_norm={delta_norm0:.8g} "
                    f"delta_ratio={delta_ratio0:.8g} "
                    f"max_abs_delta={max_abs_delta0:.8g}",
                    flush=True,
                )
                self._tgca_forward_audit_printed = True

        return out


def _apply_adapter_to_output(adapter, out):
    if torch.is_tensor(out):
        return adapter(out)

    if isinstance(out, tuple):
        out = list(out)
        for i, x in enumerate(out):
            if torch.is_tensor(x) and x.ndim == 3 and x.shape[-1] == adapter.c_out:
                out[i] = adapter(x)
                break
        return tuple(out)

    if isinstance(out, list):
        out = list(out)
        for i, x in enumerate(out):
            if torch.is_tensor(x) and x.ndim == 3 and x.shape[-1] == adapter.c_out:
                out[i] = adapter(x)
                break
        return out

    return out


def attach_tgca_if_enabled(model, args, device=None):
    if not _env_on("TGCA_MOUNT_ENABLE", "0"):
        return model

    only_model = os.environ.get("TGCA_MOUNT_ONLY_MODEL", "")
    if only_model and str(getattr(args, "model", "")) != only_model:
        return model

    root = model.module if hasattr(model, "module") else model

    # Clean post-residual protocol:
    # load a pretrained backbone checkpoint before attaching TGCA.
    load_ckpt = os.environ.get("TGCA_LOAD_BACKBONE_CKPT", "")
    if load_ckpt and not hasattr(root, "_tgca_loaded_backbone_ckpt"):
        import torch
        ckpt = torch.load(load_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        missing, unexpected = root.load_state_dict(ckpt, strict=False)
        root._tgca_loaded_backbone_ckpt = load_ckpt
        print(
            f"[TGCA_LOAD_BACKBONE] ckpt={load_ckpt} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    if hasattr(root, "tgca_adapter"):
        return model

    pred_len = int(getattr(args, "pred_len", 720))
    c_out = int(getattr(args, "c_out", getattr(args, "enc_in", 7)))
    alpha_init = float(os.environ.get("TGCA_MOUNT_ALPHA_INIT", "-8.0"))
    beta_init = float(os.environ.get("TGCA_MOUNT_BETA_INIT", "-8.0"))
    route_csv = os.environ.get("TGCA_ROUTE_INIT_CSV", "")
    dataset = str(getattr(args, "data", ""))
    backbone = str(getattr(args, "model", ""))

    root.tgca_adapter = TGCAMountedAdapter(
        pred_len=pred_len,
        c_out=c_out,
        alpha_init=alpha_init,
        beta_init=beta_init,
        route_csv=route_csv,
        dataset=dataset,
        backbone=backbone,
    )

    if device is not None:
        root.tgca_adapter.to(device)

    if _env_on("TGCA_FREEZE_BACKBONE", "0"):
        n_frozen = 0
        n_trainable = 0
        for name, param in root.named_parameters():
            if name.startswith("tgca_adapter."):
                param.requires_grad = True
                n_trainable += param.numel()
            else:
                param.requires_grad = False
                n_frozen += param.numel()
        print(
            f"[TGCA_FREEZE_BACKBONE] frozen_params={n_frozen} "
            f"trainable_tgca_params={n_trainable}",
            flush=True,
        )

    if not hasattr(root, "_tgca_original_forward"):
        root._tgca_original_forward = root.forward

        def _tgca_forward(*f_args, **f_kwargs):
            out = root._tgca_original_forward(*f_args, **f_kwargs)
            return _apply_adapter_to_output(root.tgca_adapter, out)

        root.forward = _tgca_forward

    if _env_on("TGCA_FREEZE_ADAPTER", "0"):
        for p in root.tgca_adapter.parameters():
            p.requires_grad = False
        print("[TGCA_FREEZE] adapter parameters frozen", flush=True)

    print(
        f"[TGCA_MOUNT] enabled model={backbone} pred_len={pred_len} c_out={c_out} "
        f"alpha_init={alpha_init} beta_init={beta_init} "
        f"route_loaded={getattr(root.tgca_adapter, 'route_init_loaded', 0)} "
        f"forward_patched=1",
        flush=True,
    )

    return model


def apply_tgca_if_enabled(model, outputs, args=None):
    if not _env_on("TGCA_MOUNT_ENABLE", "0"):
        return outputs
    root = model.module if hasattr(model, "module") else model
    adapter = getattr(root, "tgca_adapter", None)
    if adapter is None:
        return outputs
    return adapter(outputs)
