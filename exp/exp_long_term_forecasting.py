import os
import time
import warnings

import numpy as np
import yaml
from copy import deepcopy
import torch
import torch.nn as nn
import torch.profiler as profiler
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import MODEL_REQUIRES_CYCLE
from torch import optim
from utils.dilate_loss import dilate_loss
from utils.dilate_loss_cuda import DilateLossCUDA
from utils.tgca_latent_audit_hook import maybe_attach_latent_audit
from utils.tgca_module_shape_probe import maybe_attach_shape_probe
# from utils.dilate_loss_cache import dilate_loss
from utils.soft_dtw_cuda import SoftDTW
from utils.dtw_cuda import DTW
from utils.dpp_loss import dpp_loss
from utils.fft_ot import cal_wasserstein
from utils.fourier_koopman import fourier_loss
from utils.metrics import metric
from utils.event_weighting import build_event_weight_torch
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.hf_loss import WaveletHighFreqLoss
from utils.ot_dist import *
from utils.polynomial import chebyshev_torch, hermite_torch, laguerre_torch, leg_torch, pca_torch, Basis_Cache, ica_torch, robust_ica_torch, robust_pca_torch, svd_torch, random_torch, Random_Cache, fa_torch
from utils.tools import EarlyStopping, visual, Scheduler, adjust_learning_rate
from models.tgca_mounted_adapter import attach_tgca_if_enabled, apply_tgca_if_enabled
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len

        if args.add_noise and args.noise_amp > 0:
            seq_len = args.pred_len
            cutoff_freq_percentage = args.noise_freq_percentage
            cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
            if args.auxi_mode == "rfft":
                low_pass_mask = torch.ones(seq_len // 2 + 1)
                low_pass_mask[-cutoff_freq:] = 0.
            else:
                raise NotImplementedError
            self.mask = low_pass_mask.reshape(1, -1, 1).to(self.device)
        else:
            self.mask = None

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        exact_val = os.environ.get("TGCA_EXCHANGE_EXACT_VAL", "0") == "1"
        total_examples = 0
        self.model.eval()

        eval_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss * pred.shape[0] if exact_val else loss)
                total_examples += pred.shape[0]

        print('Validation cost time: {}'.format(time.time() - eval_time))
        # total_loss = np.average(total_loss)
        total_loss = (torch.stack(total_loss).sum() / total_examples).item() if exact_val else torch.mean(torch.stack(total_loss)).item()  # average loss
        self.model.train()
        return total_loss

    def initialize_cache(self, train_data):
        cache = None
        if self.args.auxi_mode == 'basis':
            if self.args.auxi_type == 'random':
                cache = Random_Cache(
                    rank_ratio=self.args.rank_ratio, pca_dim=self.args.pca_dim, pred_len=self.pred_len,
                    enc_in=self.args.enc_in, device=self.device
                )
            elif self.args.auxi_type == 'fa':
                cache = Basis_Cache(train_data.fa_components, train_data.initializer, mean=train_data.fa_mean, device=self.device)
            elif self.args.auxi_type == 'pca':
                cache = Basis_Cache(train_data.pca_components, train_data.initializer, weights=train_data.weights, device=self.device)
            elif self.args.auxi_type == 'robustpca':
                cache = Basis_Cache(train_data.pca_components, train_data.initializer, mean=train_data.rpca_mean, device=self.device)
            elif self.args.auxi_type == 'svd':
                cache = Basis_Cache(train_data.svd_components, train_data.initializer, device=self.device)
            elif self.args.auxi_type == 'ica':
                cache = Basis_Cache(train_data.ica_components, train_data.initializer, mean=train_data.ica_mean, whitening=train_data.whitening, device=self.device)
            elif self.args.auxi_type == 'robustica':
                cache = Basis_Cache(train_data.ica_components, train_data.initializer, device=self.device)
        return cache

    def train(self, setting, prof=None):
        train_data, train_loader = self._get_data(flag='train')
        cache = self.initialize_cache(train_data)
        vali_data, vali_loader = self._get_data(flag='val')

        train_val_start_frac = os.getenv("TGCA_TRAIN_VAL_START_FRAC")
        train_val_end_frac = os.getenv("TGCA_TRAIN_VAL_END_FRAC")
        if train_val_start_frac is not None or train_val_end_frac is not None:
            start_frac = float(train_val_start_frac or "0.0")
            end_frac = float(train_val_end_frac or "1.0")
            n_val = len(vali_data)
            start_idx = max(0, min(n_val, int(n_val * start_frac)))
            end_idx = max(start_idx + 1, min(n_val, int(n_val * end_frac)))
            vali_data = Subset(vali_data, list(range(start_idx, end_idx)))
            loader_kwargs = {
                "shuffle": False,
                "num_workers": int(self.args.num_workers),
                "drop_last": False,
                "pin_memory": getattr(vali_loader, "pin_memory", False),
                "collate_fn": getattr(vali_loader, "collate_fn", None),
            }
            vali_loader = DataLoader(
                vali_data,
                batch_size=int(getattr(self.args, "test_batch_size", self.args.batch_size)),
                **loader_kwargs,
            )
            print(
                f"[TGCA_TRAIN_VAL_SLICE] start_frac={start_frac} end_frac={end_frac} "
                f"idx={start_idx}:{end_idx}/{n_val} n={len(vali_data)}"
            )

        # ===== CPTRD_SAFE_CALIBRATION_V2 =====
        if (
            os.getenv(
                "CPTRD_CALIBRATE_ON_VAL",
                "0",
            )
            == "1"
        ):
            full_val_data = vali_data
            n_total = len(full_val_data)

            explicit = all(
                os.getenv(name) is not None
                for name in [
                    "CPTRD_CALIB_START",
                    "CPTRD_CALIB_END",
                    "CPTRD_SELECT_START",
                    "CPTRD_SELECT_END",
                ]
            )

            if explicit:
                calibration_start = int(
                    os.environ[
                        "CPTRD_CALIB_START"
                    ]
                )
                calibration_end = int(
                    os.environ[
                        "CPTRD_CALIB_END"
                    ]
                )
                selection_start = int(
                    os.environ[
                        "CPTRD_SELECT_START"
                    ]
                )
                selection_end = int(
                    os.environ[
                        "CPTRD_SELECT_END"
                    ]
                )
            else:
                calibration_start = 0

                calibration_fraction = float(
                    os.getenv(
                        "CPTRD_CALIBRATION_FRACTION",
                        "0.40",
                    )
                )

                purge_gap = int(
                    os.getenv(
                        "CPTRD_CALIBRATION_GAP",
                        str(
                            int(self.args.seq_len)
                            + int(self.args.pred_len)
                        ),
                    )
                )

                calibration_end = int(
                    n_total
                    * calibration_fraction
                )

                selection_start = (
                    calibration_end
                    + purge_gap
                )

                selection_end = n_total

            if not (
                0
                <= calibration_start
                < calibration_end
                <= selection_start
                < selection_end
                <= n_total
            ):
                raise RuntimeError(
                    "Invalid explicit split: "
                    f"total={n_total}, "
                    f"calib=[{calibration_start},"
                    f"{calibration_end}), "
                    f"select=[{selection_start},"
                    f"{selection_end})"
                )

            actual_gap = (
                selection_start
                - calibration_end
            )

            minimum_gap = int(
                os.getenv(
                    "CPTRD_MINIMUM_GAP",
                    str(
                        int(self.args.seq_len)
                        + int(self.args.pred_len)
                    ),
                )
            )

            if actual_gap < minimum_gap:
                raise RuntimeError(
                    "Temporal purge gap too small: "
                    f"{actual_gap} < {minimum_gap}"
                )

            calibration_data = Subset(
                full_val_data,
                list(
                    range(
                        calibration_start,
                        calibration_end,
                    )
                ),
            )

            selection_data = Subset(
                full_val_data,
                list(
                    range(
                        selection_start,
                        selection_end,
                    )
                ),
            )

            loader_kwargs = {
                "shuffle": False,
                "num_workers": int(
                    self.args.num_workers
                ),
                "drop_last": False,
                "pin_memory": getattr(
                    vali_loader,
                    "pin_memory",
                    False,
                ),
                "collate_fn": getattr(
                    vali_loader,
                    "collate_fn",
                    None,
                ),
            }

            train_loader = DataLoader(
                calibration_data,
                batch_size=int(
                    self.args.batch_size
                ),
                **loader_kwargs,
            )

            vali_loader = DataLoader(
                selection_data,
                batch_size=int(
                    getattr(
                        self.args,
                        "test_batch_size",
                        self.args.batch_size,
                    )
                ),
                **loader_kwargs,
            )

            train_data = calibration_data
            vali_data = selection_data

            print(
                "[CPTRD_EXPLICIT_SPLIT] "
                f"total={n_total} "
                f"calib=[{calibration_start},"
                f"{calibration_end}) "
                f"calib_n={len(calibration_data)} "
                f"gap={actual_gap} "
                f"select=[{selection_start},"
                f"{selection_end}) "
                f"select_n={len(selection_data)}"
            )
        # ===== CPTRD_SAFE_CALIBRATION_V2_END =====

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        self.writer = self._create_writer(res_path)

        time_now = time.time()

        train_steps = len(train_loader)
        model_state_last_effective = None
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        if self.args.auxi_mode == 'fourier_koopman':
            freqs = nn.Parameter(torch.tensor(train_data.freqs, device=self.device, dtype=torch.float32))
            model_optim.add_param_group({'params': freqs, 'lr': self.args.learning_rate})
        scheduler = Scheduler(model_optim, self.args, train_steps)
        criterion = self._select_criterion()
        if self.args.auxi_mode == 'soft_dtw':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "SoftDTW only supports GPU"
            sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        elif self.args.auxi_mode == 'dtw':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "DTW only supports GPU"
            dtw = DTW(use_cuda=True, bandwidth=0.1)
        elif self.args.auxi_mode == 'dilate_cuda':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "DILATE only supports GPU"
            dilate_cuda = DilateLossCUDA(alpha=self.args.dilate_alpha, gamma=self.args.gamma, bandwidth=0)

        from utils.tgca_training_resume import prepare as prepare_resume, save as save_resume
        start_epoch = prepare_resume(self, path, model_optim, scheduler, early_stopping, train_steps, vali_data, vali_loader, criterion)
        horizon_strength = float(os.environ.get('TGCA_EXCHANGE_HORIZON_WEIGHT', '0'))
        if horizon_strength < 0:
            raise ValueError('horizon weight must be nonnegative')
        if horizon_strength > 0 and (self.args.data_path != 'exchange_rate.csv' or self.args.loss.lower() != 'mse'):
            raise ValueError('Exchange horizon weighting requires Exchange data and MSE loss')
        def reconstruction_criterion(prediction, target):
            if horizon_strength == 0:
                return criterion(prediction, target)
            weights = 1.0 + horizon_strength * torch.linspace(0, 1, prediction.shape[1], device=prediction.device, dtype=prediction.dtype)
            weights = weights / weights.mean()
            return ((prediction - target).pow(2) * weights.view(1, -1, 1)).mean()
        if horizon_strength:
            print(f'[TGCA_EXCHANGE_HORIZON_WEIGHT] strength={horizon_strength}', flush=True)

        for epoch in range(start_epoch, self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            has_nan_in_epoch = False
            train_loss = []

            lr_cur = scheduler.get_lr()
            lr_cur = lr_cur[0] if isinstance(lr_cur, list) else lr_cur
            self.writer.add_scalar(f'{self.pred_len}/train/lr', lr_cur, self.epoch)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                outputs, batch_y, attn = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                loss = 0
                if self.args.rec_lambda:
                    if prof is not None:
                        with profiler.record_function("rec_loss_forward_pass"):
                            loss_rec = reconstruction_criterion(outputs, batch_y)
                    else:
                        loss_rec = reconstruction_criterion(outputs, batch_y)
                    loss += self.args.rec_lambda * loss_rec
                else:
                    loss_rec = torch.tensor(1e4)
                if self.step % self.log_step == 0:
                    self.writer.add_scalar(f'{self.pred_len}/train/loss_rec', loss_rec, self.step)

                if self.args.l1_weight and attn:
                    loss += self.args.l1_weight * attn[0]

                if self.args.auxi_lambda:
                    if self.args.joint_forecast:  # joint distribution forecasting
                        outputs = torch.concat((batch_x.to(outputs.device), outputs), dim=1)  # [B, S+P, D]
                        batch_y = torch.concat((batch_x.to(batch_y.device), batch_y), dim=1)  # [B, S+P, D]

                    if self.args.auxi_mode == "fft":
                        loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)  # shape: [B, P, D]

                    elif self.args.auxi_mode == "rfft":
                        if self.args.auxi_type == 'complex':
                            loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)  # shape: [B, P//2+1, D]
                        elif self.args.auxi_type == 'complex-phase':
                            loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                        elif self.args.auxi_type == 'complex-mag-phase':
                            loss_auxi_mag = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs()
                            loss_auxi_phase = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])  # shape: [2, B, P//2+1, D]
                        elif self.args.auxi_type == 'phase':
                            loss_auxi = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()  # shape: [B, P//2+1, D]
                        elif self.args.auxi_type == 'mag':
                            loss_auxi = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()  # shape: [B, P//2+1, D]
                        elif self.args.auxi_type == 'mag-phase':
                            loss_auxi_mag = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                            loss_auxi_phase = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
                            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])  # shape: [2, B, P//2+1, D]
                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "rfft-D":
                        loss_auxi = torch.fft.rfft(outputs, dim=-1) - torch.fft.rfft(batch_y, dim=-1)  # shape: [B, P, D//2+1]

                    elif self.args.auxi_mode == "rfft-2D":
                        loss_auxi = torch.fft.rfft2(outputs) - torch.fft.rfft2(batch_y)  # shape: [B, P, D//2+1]

                    elif self.args.auxi_mode == "basis":
                        kwargs = {'degree': self.args.leg_degree, 'device': self.device}
                        if self.args.auxi_type == "legendre":
                            loss_auxi = leg_torch(outputs, **kwargs) - leg_torch(batch_y, **kwargs)  # shape: [B*D, degree+1]
                        elif self.args.auxi_type == "chebyshev":
                            loss_auxi = chebyshev_torch(outputs, **kwargs) - chebyshev_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "hermite":
                            loss_auxi = hermite_torch(outputs, **kwargs) - hermite_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "laguerre":
                            loss_auxi = laguerre_torch(outputs, **kwargs) - laguerre_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "random":
                            kwargs = {'pca_dim': self.args.pca_dim, 'random_cache': cache, 'device': self.device}
                            loss_auxi = random_torch(outputs, **kwargs) - random_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "fa":
                            kwargs = {'pca_dim': self.args.pca_dim, 'fa_cache': cache, 'reinit': self.args.reinit, 'device': self.device}
                            loss_auxi = fa_torch(outputs, **kwargs) - fa_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "pca":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'pca_cache': cache, 'use_weights': self.args.use_weights,
                                'reinit': self.args.reinit, 'device': self.device
                            }
                            if prof is not None:
                                with profiler.record_function("auxi_loss_forward_pass"):
                                    loss_auxi = pca_torch(outputs, **kwargs) - pca_torch(batch_y, **kwargs)
                            else:
                                loss_auxi = pca_torch(outputs, **kwargs) - pca_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "robustpca":
                            kwargs = {'pca_dim': self.args.pca_dim, 'pca_cache': cache, 'reinit': self.args.reinit, 'device': self.device}
                            loss_auxi = robust_pca_torch(outputs, **kwargs) - robust_pca_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "svd":
                            kwargs = {'pca_dim': self.args.pca_dim, 'svd_cache': cache, 'reinit': self.args.reinit, 'device': self.device}
                            loss_auxi = svd_torch(outputs, **kwargs) - svd_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "ica":
                            kwargs = {'pca_dim': self.args.pca_dim, 'ica_cache': cache, 'reinit': self.args.reinit, 'device': self.device}
                            loss_auxi = ica_torch(outputs, **kwargs) - ica_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "robustica":
                            kwargs = {'pca_dim': self.args.pca_dim, 'ica_cache': cache, 'reinit': self.args.reinit, 'device': self.device}
                            loss_auxi = robust_ica_torch(outputs, **kwargs) - robust_ica_torch(batch_y, **kwargs)
                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "ot":
                        kwargs = {'dist_scale': self.args.dist_scale, 'device': self.device}
                        if self.args.auxi_type == "emd1d_t":
                            loss_auxi = emd_loss_1d_batched_align_t(outputs, batch_y, **kwargs)
                        elif self.args.auxi_type == "emd1d_h":
                            loss_auxi = emd_loss_1d_batched_align_h(outputs, batch_y, **kwargs)
                        elif self.args.auxi_type == "emd1d_all":
                            loss_auxi = emd_loss_1d_batched_align_all(outputs, batch_y, **kwargs)

                        elif self.args.auxi_type == "emd2d_h":
                            loss_auxi = emd_loss_2d_batched_align_h(outputs, batch_y, **kwargs)
                        elif self.args.auxi_type == "emd2d_t":
                            loss_auxi = emd_loss_2d_batched_align_t(outputs, batch_y, **kwargs)
                        elif self.args.auxi_type == "emd2d_all":
                            loss_auxi = emd_loss_2d_batched_align_all(outputs, batch_y, **kwargs)

                        elif self.args.auxi_type == "emd1d_h_learn_proj":
                            outputs_proj = self.model.project(outputs)
                            batch_y_proj = self.model.project(batch_y)
                            loss_auxi = emd_loss_1d_batched_align_h(outputs_proj, batch_y_proj, **kwargs)
                        elif self.args.auxi_type == "emd1d_t_learn_proj":
                            outputs_proj = self.model.project(outputs)
                            batch_y_proj = self.model.project(batch_y)
                            loss_auxi = emd_loss_1d_batched_align_t(outputs_proj, batch_y_proj, **kwargs)
                        elif self.args.auxi_type == "emd1d_all_learn_proj":
                            outputs_proj = self.model.project(outputs)
                            batch_y_proj = self.model.project(batch_y)
                            loss_auxi = emd_loss_1d_batched_align_all(outputs_proj, batch_y_proj, **kwargs)

                        elif self.args.auxi_type == "emd1d_h_pca_proj":
                            n_feats, rank_ratio = self.args.c_out, self.args.rank_ratio
                            low_rank = int(n_feats * rank_ratio)
                            outputs_proj = torch.matmul(outputs, torch.pca_lowrank(outputs.reshape(-1, n_feats), low_rank)[-1])
                            batch_y_proj = torch.matmul(batch_y, torch.pca_lowrank(batch_y.reshape(-1, n_feats), low_rank)[-1])
                            loss_auxi = emd_loss_1d_batched_align_h(outputs_proj, batch_y_proj, **kwargs)
                        elif self.args.auxi_type == "emd1d_t_pca_proj":
                            n_feats, rank_ratio = self.args.c_out, self.args.rank_ratio
                            low_rank = int(n_feats * rank_ratio)
                            outputs_proj = torch.matmul(outputs, torch.pca_lowrank(outputs.reshape(-1, n_feats), low_rank)[-1])
                            batch_y_proj = torch.matmul(batch_y, torch.pca_lowrank(batch_y.reshape(-1, n_feats), low_rank)[-1])
                            loss_auxi = emd_loss_1d_batched_align_t(outputs_proj, batch_y_proj, **kwargs)
                        elif self.args.auxi_type == "emd1d_all_pca_proj":
                            n_feats, rank_ratio = self.args.c_out, self.args.rank_ratio
                            low_rank = int(n_feats * rank_ratio)
                            outputs_proj = torch.matmul(outputs, torch.pca_lowrank(outputs.reshape(-1, n_feats), low_rank)[-1])
                            batch_y_proj = torch.matmul(batch_y, torch.pca_lowrank(batch_y.reshape(-1, n_feats), low_rank)[-1])
                            loss_auxi = emd_loss_1d_batched_align_all(outputs_proj, batch_y_proj, **kwargs)

                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "fft_ot":
                        loss_auxi = cal_wasserstein(
                            outputs, batch_y, self.args.distance, ot_type=self.args.ot_type, normalize=self.args.normalize,
                            mask_factor=self.args.mask_factor, reg_sk=self.args.reg_sk, stopThr=self.args.stopThr, numItermax=self.args.numItermax,
                            var_weight=self.args.var_weight, mean_weight=self.args.mean_weight
                        )

                    elif self.args.auxi_mode == "fourier_koopman":
                        loss_auxi = fourier_loss(outputs, batch_y, freqs, device=self.device)

                    elif self.args.auxi_mode == "dilate":
                        loss_auxi, _, _ = dilate_loss(outputs, batch_y, self.args.alpha, self.args.gamma, self.device)

                    elif self.args.auxi_mode == "dpp":
                        loss_auxi = dpp_loss(outputs, batch_y, self.args.alpha, self.args.gamma, self.device)

                    elif self.args.auxi_mode == "soft_dtw":
                        loss_auxi = sdtw(outputs, batch_y)

                    elif self.args.auxi_mode == "dtw":
                        loss_auxi = dtw(outputs, batch_y)[0].mean()

                    elif self.args.auxi_mode == "dilate_cuda":
                        loss_auxi = dilate_cuda(outputs, batch_y)

                    else:
                        raise NotImplementedError

                    if self.mask is not None:
                        loss_auxi *= self.mask

                    if self.args.auxi_loss == "MAE":
                        # MAE, 最小化element-wise error的模长
                        loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
                    elif self.args.auxi_loss == "MSE":
                        # MSE, 最小化element-wise error的模长
                        loss_auxi = (loss_auxi.abs()**2).mean() if self.args.module_first else (loss_auxi**2).mean().abs()
                    elif self.args.auxi_loss == "None":
                        pass
                    else:
                        raise NotImplementedError

                    loss_hf = torch.tensor(0.0, device=outputs.device)
                    if getattr(self.args, 'use_hf_loss', 0) == 1 and getattr(self.args, 'hf_lambda', 0.0) > 0:
                        loss_hf = self.hf_loss_fn(outputs, batch_y)

                    loss += self.args.auxi_lambda * loss_auxi + getattr(self.args, 'hf_lambda', 0.0) * loss_hf
                else:
                    loss_auxi = torch.tensor(1e4, device=outputs.device)
                    loss_hf = torch.tensor(0.0, device=outputs.device)
                if self.step % self.log_step == 0:
                    self.writer.add_scalar(f'{self.pred_len}/train/loss_auxi', loss_auxi, self.step)

                # ===== CPTRD_DIRECTIONAL_V2_BEGIN =====
                cptrd_model = (
                    self.model.module
                    if hasattr(self.model, "module")
                    else self.model
                )

                if hasattr(
                    cptrd_model,
                    "cptrd_supervision_loss",
                ):
                    (
                        cptrd_directional_loss,
                        cptrd_directional_stats,
                    ) = cptrd_model.cptrd_supervision_loss(
                        batch_y
                    )

                    loss = (
                        loss
                        + cptrd_directional_loss
                    )

                    if (i + 1) % 100 == 0:
                        stat_text = " ".join(
                            f"{key}={float(value):.6g}"
                            for key, value
                            in cptrd_directional_stats.items()
                        )

                        print(
                            "[CPTRD_DIRECTION] "
                            f"loss="
                            f"{float(cptrd_directional_loss):.6g} "
                            + stat_text
                        )
                # ===== CPTRD_DIRECTIONAL_V2_END =====

                if not torch.is_tensor(loss):
                    raise RuntimeError(
                        "Training loss is not a Tensor. "
                        "All enabled loss weights may be zero. "
                        f"loss_type={type(loss).__name__}, "
                        f"rec_lambda={getattr(self.args, 'rec_lambda', None)}, "
                        f"auxi_lambda={getattr(self.args, 'auxi_lambda', None)}"
                    )

                if not torch.isfinite(loss).all():
                    print(f"Loss is NaN or Inf, skipping epoch {self.epoch} step {self.step}")
                    has_nan_in_epoch = True
                    continue

                train_loss.append(loss.item())
                self.writer.add_scalar(f'{self.pred_len}/train/loss_iter', loss.item(), self.step)

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {}, epoch: {} | loss_rec: {:.7f}, loss_auxi: {:.7f}, loss_hf: {:.7f}, loss: {:.7f}".format(
                            i + 1, self.epoch, loss_rec.item(), loss_auxi.item(), loss_hf.item(), loss.item()
                        )
                    )
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                    iter_count = 0
                    time_now = time.time()
                    model_state_last_effective = deepcopy(self.model.state_dict())  # save the last effective model state dict

                if prof is not None:
                    with profiler.record_function("backward_pass"):
                        loss.backward()
                else:
                    loss.backward()
                model_optim.step()

                if prof is not None:
                    prof.step()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

            if model_state_last_effective is not None and has_nan_in_epoch:
                self.model.load_state_dict(model_state_last_effective)

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar(f'{self.pred_len}/train/loss', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss', vali_loss, self.epoch)

            print(
                "Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                    self.epoch, self.step, train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if not early_stopping.early_stop and self.args.lradj not in ['TST']:
                scheduler.step(vali_loss, self.epoch)
            save_resume(self, path, model_optim, scheduler, early_stopping, train_steps, self.epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def _test_lowmem_streaming(self, setting, test_data, test_loader, folder_path, prof=None):
        import os
        import yaml
        import numpy as np
        import torch

        print("[LOWMEM_STREAM_TEST] enabled: streaming metrics only; pred.npy/true.npy/input.npy will NOT be saved")

        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)

        if self.writer is None:
            self.writer = self._create_writer(res_path)

        self.model.eval()

        start_frac = float(os.environ.get("TGCA_STREAM_START_FRAC", "0.0"))
        end_frac = float(os.environ.get("TGCA_STREAM_END_FRAC", "1.0"))
        start_frac = min(max(start_frac, 0.0), 1.0)
        end_frac = min(max(end_frac, start_frac), 1.0)
        total_batches = len(test_loader)
        start_batch = int(total_batches * start_frac)
        end_batch = int(total_batches * end_frac)
        if end_batch <= start_batch and total_batches > 0:
            end_batch = start_batch + 1
        if start_batch > 0 or end_batch < total_batches:
            print(
                "[LOWMEM_STREAM_SLICE] "
                f"start_frac={start_frac} end_frac={end_frac} "
                f"batches={start_batch}:{end_batch}/{total_batches}",
                flush=True,
            )

        eps = 1e-5
        n_elem = 0
        se_sum = 0.0
        ae_sum = 0.0
        ape_sum = 0.0
        spe_sum = 0.0
        true_abs_sum = 0.0

        first_shape = None
        last_shape = None

        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad

        with ctx():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                if i < start_batch:
                    continue
                if i >= end_batch:
                    break

                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                batch_x = batch_x.detach()
                outputs = outputs.detach()
                batch_y = batch_y.detach()

                if test_data.scale and self.args.inverse:
                    batch_x_np = batch_x.cpu().numpy()
                    in_shape = batch_x_np.shape
                    batch_x_np = test_data.inverse_transform(batch_x_np.reshape(-1, in_shape[-1])).reshape(in_shape)
                    batch_x = torch.from_numpy(batch_x_np).float().to(self.device)

                    outputs_np = outputs.cpu().numpy()
                    batch_y_np = batch_y.cpu().numpy()
                    out_shape = outputs_np.shape
                    outputs_np = test_data.inverse_transform(outputs_np.reshape(-1, out_shape[-1])).reshape(out_shape)
                    batch_y_np = test_data.inverse_transform(batch_y_np.reshape(-1, out_shape[-1])).reshape(out_shape)
                    outputs = torch.from_numpy(outputs_np).float().to(self.device)
                    batch_y = torch.from_numpy(batch_y_np).float().to(self.device)

                if first_shape is None:
                    first_shape = tuple(outputs.shape)
                last_shape = tuple(outputs.shape)

                err = outputs - batch_y
                abs_err = torch.abs(err)
                abs_true = torch.abs(batch_y)
                denom = torch.clamp(abs_true, min=eps)

                se_sum += torch.sum(err * err).item()
                ae_sum += torch.sum(abs_err).item()
                ape_sum += torch.sum(abs_err / denom).item()
                spe_sum += torch.sum((err / denom) ** 2).item()
                true_abs_sum += torch.sum(abs_true).item()
                n_elem += err.numel()

                if i % 20 == 0 and self.output_vis:
                    gt = np.concatenate((batch_x[0, :, -1].cpu().numpy(), batch_y[0, :, -1].cpu().numpy()), axis=0)
                    pd = np.concatenate((batch_x[0, :, -1].cpu().numpy(), outputs[0, :, -1].cpu().numpy()), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                if i % 50 == 0:
                    print(f"[LOWMEM_STREAM_TEST] batch={i}, n_elem={n_elem}, first_shape={first_shape}, last_shape={last_shape}", flush=True)

                if prof is not None:
                    prof.step()

                del batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, outputs, err, abs_err, abs_true, denom

        mse = se_sum / max(1, n_elem)
        mae = ae_sum / max(1, n_elem)
        rmse = float(np.sqrt(mse))
        mape = ape_sum / max(1, n_elem)
        mspe = spe_sum / max(1, n_elem)
        mre = ae_sum / max(eps, true_abs_sum)

        print("test shape streaming: first={}, last={}, n_elem={}".format(first_shape, last_shape, n_elem))
        print('{}\t| mse:{}, mae:{}'.format(self.pred_len, mse, mae))

        # ===== FORCE_SAVE_PRED_TRUE_START =====
        if getattr(self.args, "output_pred", False) or os.environ.get("TGCA_FORCE_SAVE_PREDTRUE", "0") == "1":
            from pathlib import Path
            save_dir = Path(os.environ.get(
                "TGCA_FORCE_SAVE_DIR",
                os.path.join(self.args.test_results, setting)
            ))
            save_dir.mkdir(parents=True, exist_ok=True)

            safe_setting = setting.replace("/", "__")
            pred_path = save_dir / f"{safe_setting}_pred.npy"
            true_path = save_dir / f"{safe_setting}_true.npy"
            input_path = save_dir / f"{safe_setting}_input.npy"
            metric_path = save_dir / f"{safe_setting}_metrics.csv"

            np.save(pred_path, preds.cpu().numpy())
            np.save(true_path, trues.cpu().numpy())
            np.save(input_path, inputs.cpu().numpy())

            with open(metric_path, "w") as f:
                f.write("pred_len,mse,mae,rmse,mape,mspe,mre\n")
                f.write(f"{self.pred_len},{float(mse)},{float(mae)},{float(rmse)},{float(mape)},{float(mspe)},{float(mre)}\n")

            print(f"[FORCE_SAVE_PRED_TRUE] pred={pred_path}", flush=True)
            print(f"[FORCE_SAVE_PRED_TRUE] true={true_path}", flush=True)
            print(f"[FORCE_SAVE_PRED_TRUE] input={input_path}", flush=True)
            print(f"[FORCE_SAVE_PRED_TRUE] metrics={metric_path}", flush=True)
        # ===== FORCE_SAVE_PRED_TRUE_END =====

        if os.environ.get("TGCA_FORCE_SAVE_PREDTRUE", "0") == "1":
            from pathlib import Path
            save_dir = Path(os.environ.get(
                "TGCA_FORCE_SAVE_DIR",
                "experiments/tgca_forecaster_export_from_ckpt_v1/predtrue"
            ))
            save_dir.mkdir(parents=True, exist_ok=True)

            safe_setting = setting.replace("/", "__")
            pred_path = save_dir / f"{safe_setting}_pred.npy"
            true_path = save_dir / f"{safe_setting}_true.npy"
            input_path = save_dir / f"{safe_setting}_input.npy"
            metric_path = save_dir / f"{safe_setting}_metrics.csv"

            np.save(pred_path, preds.cpu().numpy())
            np.save(true_path, trues.cpu().numpy())
            np.save(input_path, inputs.cpu().numpy())

            with open(metric_path, "w") as f:
                f.write("pred_len,mse,mae,rmse,mape,mspe,mre\n")
                f.write(f"{self.pred_len},{float(mse)},{float(mae)},{float(rmse)},{float(mape)},{float(mspe)},{float(mre)}\n")

            print(f"[TGCA_FORCE_SAVE_PREDTRUE] pred={pred_path}", flush=True)
            print(f"[TGCA_FORCE_SAVE_PREDTRUE] true={true_path}", flush=True)
            print(f"[TGCA_FORCE_SAVE_PREDTRUE] input={input_path}", flush=True)
            print(f"[TGCA_FORCE_SAVE_PREDTRUE] metrics={metric_path}", flush=True)

        self.writer.add_scalar(f'{self.pred_len}/test/mae', mae, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mse', mse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/rmse', rmse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mape', mape, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mspe', mspe, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mre', mre, self.epoch)
        self.writer.close()

        log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
        with open(log_path, 'a') as f:
            f.write(setting + "\n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n\n')

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, mre]))

        with open(os.path.join(res_path, "lowmem_stream_metrics.txt"), "w") as f:
            f.write(f"setting={setting}\n")
            f.write(f"pred_len={self.pred_len}\n")
            f.write(f"first_shape={first_shape}\n")
            f.write(f"last_shape={last_shape}\n")
            f.write(f"n_elem={n_elem}\n")
            f.write(f"mse={mse}\n")
            f.write(f"mae={mae}\n")
            f.write(f"rmse={rmse}\n")
            f.write(f"mape={mape}\n")
            f.write(f"mspe={mspe}\n")
            f.write(f"mre={mre}\n")

        if not os.path.exists(os.path.join(res_path, 'config.yaml')):
            print('save configs')
            args_dict = vars(self.args)
            with open(os.path.join(res_path, 'config.yaml'), 'w') as yaml_file:
                yaml.dump(args_dict, yaml_file, default_flow_style=False)

        return


    def _test_line_only_export(self, setting, test_data, test_loader, folder_path, prof=None):
        """
        Export only selected sample/channel horizon lines for plotting.
        Env:
          LINE_ONLY_EXPORT=1
          LINE_SAMPLES=0,10
          LINE_CHANNELS=0,6
        Output:
          <results>/<setting>/line_only_exports/sample{idx}_ch{ch}.npz
        """
        samples = [int(x) for x in os.environ.get("LINE_SAMPLES", "0,10").split(",") if x.strip() != ""]
        channels = [int(x) for x in os.environ.get("LINE_CHANNELS", "0,6").split(",") if x.strip() != ""]
        samples = sorted(set(samples))
        channels = sorted(set(channels))

        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        line_dir = os.path.join(res_path, "line_only_exports")
        os.makedirs(line_dir, exist_ok=True)

        manifest = []
        wanted = set((si, ch) for si in samples for ch in channels)
        saved = set()
        seen = 0
        max_sample = max(samples) if samples else 0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                batch_x = batch_x.detach()
                outputs = outputs.detach()
                batch_y = batch_y.detach()

                if test_data.scale and self.args.inverse:
                    outputs_np = outputs.cpu().numpy()
                    batch_y_np = batch_y.cpu().numpy()
                    out_shape = outputs_np.shape
                    outputs_np = test_data.inverse_transform(outputs_np.reshape(-1, out_shape[-1])).reshape(out_shape)
                    batch_y_np = test_data.inverse_transform(batch_y_np.reshape(-1, out_shape[-1])).reshape(out_shape)
                    outputs = torch.from_numpy(outputs_np).float()
                    batch_y = torch.from_numpy(batch_y_np).float()

                bsz = outputs.shape[0]
                cdim = outputs.shape[-1]

                for local_i in range(bsz):
                    global_i = seen + local_i
                    if global_i not in samples:
                        continue

                    for ch in channels:
                        if ch < 0 or ch >= cdim:
                            continue

                        pred_line = outputs[local_i, :, ch].detach().cpu().numpy()
                        true_line = batch_y[local_i, :, ch].detach().cpu().numpy()

                        out_file = os.path.join(line_dir, f"sample{global_i}_ch{ch}.npz")
                        np.savez_compressed(
                            out_file,
                            pred=pred_line,
                            true=true_line,
                            sample=np.array(global_i),
                            channel=np.array(ch),
                        )

                        mse = float(np.mean((pred_line.astype(np.float64) - true_line.astype(np.float64)) ** 2))
                        mae = float(np.mean(np.abs(pred_line.astype(np.float64) - true_line.astype(np.float64))))
                        manifest.append((global_i, ch, out_file, mse, mae))
                        saved.add((global_i, ch))
                        print(f"[LINE_ONLY_EXPORT] saved sample={global_i} ch={ch} mse={mse:.6f} mae={mae:.6f} -> {out_file}", flush=True)

                seen += bsz

                if seen > max_sample and wanted.issubset(saved):
                    break

                if prof is not None:
                    prof.step()

        manifest_path = os.path.join(line_dir, "manifest.csv")
        with open(manifest_path, "w") as f:
            f.write("sample,channel,file,mse,mae\n")
            for row in manifest:
                f.write("{},{},{},{},{}\n".format(*row))

        print(f"[LINE_ONLY_EXPORT] done. saved={len(manifest)} manifest={manifest_path}", flush=True)
        return

    def test(self, setting, prof=None, test=0):
        import os
        export_flag = os.environ.get('TGCA_EXPORT_SPLIT_FLAG', 'test')
        print(f'[TGCA_EXPORT_SPLIT] flag={export_flag}', flush=True)
        test_data, test_loader = self._get_data(flag=export_flag)
        if (
            str(export_flag).lower() in {"val", "test"}
            and os.environ.get(
                "TGCA_ORDERED_VAL_EXPORT",
                "0",
            ) == "1"
        ):
            from torch.utils.data import DataLoader

            export_batch_size = int(
                getattr(
                    self.args,
                    "test_batch_size",
                    getattr(self.args, "batch_size", 1),
                )
            )

            test_loader = DataLoader(
                test_data,
                batch_size=export_batch_size,
                shuffle=False,
                num_workers=int(
                    getattr(self.args, "num_workers", 0)
                ),
                drop_last=False,
            )

            print(
                "[TGCA_ORDERED_VAL_EXPORT] "
                f"n={len(test_data)} "
                f"batch_size={export_batch_size} "
                "shuffle=False drop_last=False",
                flush=True,
            )
        if test:
            print('loading model')
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            export_load_ckpt = os.environ.get('TGCA_EXPORT_LOAD_CKPT', '')
            load_path = export_load_ckpt if export_load_ckpt else os.path.join(ckpt_dir, 'checkpoint.pth')
            # ===== TGCA_TEST_USE_CURRENT_MODEL_V1 =====
            use_current_model = (
                os.environ.get(
                    "TGCA_TEST_USE_CURRENT_MODEL",
                    "0",
                ) == "1"
            )

            if use_current_model:
                if not getattr(
                    self.model,
                    "_tgca_train_checkpoint_loaded",
                    False,
                ):
                    root_model = getattr(
                        self.model,
                        "module",
                        self.model,
                    )

                    loaded = getattr(
                        root_model,
                        "_tgca_train_checkpoint_loaded",
                        False,
                    )
                else:
                    loaded = True

                print(
                    "[TGCA_TEST_USE_CURRENT_MODEL]",
                    {
                        "skip_disk_reload": True,
                        "train_checkpoint_loaded":
                            bool(loaded),
                        "fallback_path_not_used":
                            str(load_path),
                    },
                    flush=True,
                )

                if not loaded:
                    raise RuntimeError(
                        "TGCA_TEST_USE_CURRENT_MODEL=1, "
                        "but no training checkpoint load marker "
                        "was found on the model"
                    )
            else:
                print(
                    f"[TGCA_EXPORT_LOAD_CKPT] {load_path}",
                    flush=True,
                )

                self.model.load_state_dict(
                    torch.load(
                        load_path,
                        map_location=torch.device(
                            "cpu"
                        ),
                    )
                )
            # ===== TGCA_TEST_USE_CURRENT_MODEL_V1_END =====

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        os.makedirs(folder_path, exist_ok=True)

        if os.environ.get("LOWMEM_STREAM_TEST", "0") == "1":
            return self._test_lowmem_streaming(setting, test_data, test_loader, folder_path, prof=prof)

        if os.environ.get("LINE_ONLY_EXPORT", "0") == "1":
            return self._test_line_only_export(setting, test_data, test_loader, folder_path, prof=prof)

        self.model.eval()
        maybe_attach_latent_audit(self.model)
        maybe_attach_shape_probe(self.model)
        # metric_collector = create_metric_collector(device=self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                batch_x = batch_x.detach()
                outputs = outputs.detach()
                batch_y = batch_y.detach()

                if test_data.scale and self.args.inverse:
                    batch_x = batch_x.cpu().numpy()
                    in_shape = batch_x.shape
                    batch_x = test_data.inverse_transform(batch_x.reshape(-1, in_shape[-1])).reshape(in_shape)
                    batch_x = torch.from_numpy(batch_x).float().to(self.device)

                    outputs = outputs.cpu().numpy()
                    batch_y = batch_y.cpu().numpy()
                    out_shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, out_shape[-1])).reshape(out_shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, out_shape[-1])).reshape(out_shape)
                    outputs = torch.from_numpy(outputs).float().to(self.device)
                    batch_y = torch.from_numpy(batch_y).float().to(self.device)

                inputs.append(batch_x.cpu())
                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())

                if i % 20 == 0 and self.output_vis:
                    gt = np.concatenate((batch_x[0, :, -1].cpu().numpy(), batch_y[0, :, -1].cpu().numpy()), axis=0)
                    pd = np.concatenate((batch_x[0, :, -1].cpu().numpy(), outputs[0, :, -1].cpu().numpy()), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                if prof is not None:
                    prof.step()

        inputs = torch.cat(inputs, dim=0)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        print('test shape:', preds.shape, trues.shape)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        # m = metric_collector.compute()
        # mae, mse, rmse, mape, mspe, mre = m["mae"], m["mse"], m["rmse"], m["mape"], m["mspe"], m["mre"]
        mae, mse, rmse, mape, mspe, mre = metric_torch(preds, trues)
        print('{}\t| mse:{}, mae:{}'.format(self.pred_len, mse, mae))

        self.writer.add_scalar(f'{self.pred_len}/test/mae', mae, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mse', mse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/rmse', rmse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mape', mape, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mspe', mspe, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mre', mre, self.epoch)
        self.writer.close()

        log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
        f = open(log_path, 'a')
        f.write(setting + "\n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, mre]))

        if self.output_pred or os.environ.get("FORCE_SAVE_PREDTRUE", "0") == "1":
            # LOWDISK: skip saving input.npy / val_input.npy
            if os.environ.get('TGCA_SKIP_SAVE_PRED', '0') != '1':
                np.save(os.path.join(res_path, 'val_pred.npy' if os.environ.get('VAL_EXPORT', '0') == '1' else 'pred.npy'), preds.cpu().numpy())
            if os.environ.get('SKIP_TRUE_NPY', '0') != '1':
                np.save(os.path.join(res_path, 'val_true.npy' if os.environ.get('VAL_EXPORT', '0') == '1' else 'true.npy'), trues.cpu().numpy())
            if self.args.auxi_mode == 'basis' and self.args.auxi_type == 'pca':
                train_data, _ = self._get_data(flag='train')
                pca_components = train_data.pca_components
                np.save(os.path.join(res_path, 'pca_components.npy'), pca_components)

        if not test or not os.path.exists(os.path.join(res_path, 'config.yaml')):
            print('save configs')
            args_dict = vars(self.args)
            with open(os.path.join(res_path, 'config.yaml'), 'w') as yaml_file:
                yaml.dump(args_dict, yaml_file, default_flow_style=False)

        return
