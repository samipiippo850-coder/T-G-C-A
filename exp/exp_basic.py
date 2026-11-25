import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from data_provider.data_factory import data_provider
from models import MODEL_DICT, MODEL_REQUIRES_CYCLE
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def add_scalars(self, *args, **kwargs): pass
        def add_histogram(self, *args, **kwargs): pass
        def add_text(self, *args, **kwargs): pass
        def flush(self): pass
        def close(self): pass
    print("[WARN] tensorboard is not installed; using no-op SummaryWriter")
from utils.tools import pv
from models.tgca_mounted_adapter import attach_tgca_if_enabled, apply_tgca_if_enabled


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = MODEL_DICT
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.model = attach_tgca_if_enabled(self.model, self.args, getattr(self, 'device', None))
        # ===== TGCA_TRAIN_CHECKPOINT_LOADER_V1 =====
        from utils.tgca_train_checkpoint import maybe_load_train_checkpoint
        self.model = maybe_load_train_checkpoint(
            self.model,
            getattr(self, 'device', None),
        )
        # ===== TGCA_TRAIN_CHECKPOINT_LOADER_V1_END =====
        pv(self.model, args.verbose)
        self.writer = None
        self.log_step = args.log_step

        self.epoch = 0
        self.step = 0

        self.output_pred = args.output_pred
        self.output_vis = args.output_vis

    def _build_model(self):
        if self.args.model == 'TGCA_Forecaster':
            import importlib
            TGCA_Forecaster = importlib.import_module('models.TGCA_Forecaster')
            model = TGCA_Forecaster.Model(self.args).float()
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()

        pretrain_model_path = self.args.pretrain_model_path
        if pretrain_model_path and os.path.exists(pretrain_model_path):
            print(f'Loading pretrained model from {pretrain_model_path}')
            state_dict = torch.load(pretrain_model_path)
            model.load_state_dict(state_dict, strict=False)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) \
                if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _create_writer(self, log_dir):
        item_list = os.listdir(log_dir)
        item_path_list = [os.path.join(log_dir, item) for item in item_list]
        item_path_list = [item_path for item_path in item_path_list if os.path.isfile(item_path)]
        if len(item_path_list) > 0:
            pre_log_dir = os.path.join(log_dir, "pre_logs")
            os.makedirs(pre_log_dir, exist_ok=True)

            item_list = [os.path.basename(item_path) for item_path in item_path_list]
            for item, item_path in zip(item_list, item_path_list):
                shutil.move(item_path, os.path.join(pre_log_dir, item))

        return SummaryWriter(log_dir)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, model=None, lr=None, optim_type=None):
        if model is None:
            model = self.model
        if lr is None:
            lr = self.args.learning_rate
        if optim_type is None:
            optim_type = self.args.optim_type
        if optim_type == 'adam':
            optim_class = optim.Adam
        elif optim_type == 'adamw':
            optim_class = optim.AdamW
        elif optim_type == 'sgd':
            optim_class = optim.SGD
        # ===== TGCA_WEIGHT_DECAY_V1 =====
        # AdamW has always been created WITHOUT weight_decay so far, i.e. torch's default 0.01 was in
        # effect. --cf_weight_decay now lets a run override it explicitly (0 keeps the old behaviour).
        weight_decay = float(getattr(self.args, 'cf_weight_decay', 0) or 0)
        if weight_decay <= 0:
            weight_decay = 0.01
        # ===== TGCA_SPLIT_LR_V1 =====
        split = self._split_base_adapter_groups(model, lr)
        if split is not None:
            model_optim = optim_class(
                [{'params': params, 'lr': group_lr} for _, params, group_lr in split],
                weight_decay=weight_decay,
            )
            summary = ', '.join(
                f'{name}_lr={group_lr} {name}_params={sum(p.numel() for p in params)}'
                for name, params, group_lr in split
            )
            print(f'[TGCA_SPLIT_LR] {summary} weight_decay={weight_decay}', flush=True)
            return model_optim
        # ===== TGCA_SPLIT_LR_V1_END =====
        model_optim = optim_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        return model_optim

    # ===== TGCA_SPLIT_LR_V1 =====
    def _split_base_adapter_groups(self, model, fallback_lr):
        """Split TGCA parameters into base / adapter optimizer groups.

        Returns None when the model exposes no separate ``base`` / ``adapter``
        submodules or when neither ``--base_learning_rate`` nor
        ``--adapter_learning_rate`` is given, so the legacy single-lr path is
        untouched. Parameters that belong to neither branch (if any) follow the
        adapter learning rate.
        """
        base_lr = getattr(self.args, 'base_learning_rate', None)
        adapter_lr = getattr(self.args, 'adapter_learning_rate', None)
        if base_lr is None and adapter_lr is None:
            return None
        core = getattr(model, 'module', model)
        base_module = getattr(core, 'base', None)
        adapter_module = getattr(core, 'adapter', None)
        if base_module is None or adapter_module is None:
            return None
        base_lr = float(fallback_lr if base_lr is None else base_lr)
        adapter_lr = float(fallback_lr if adapter_lr is None else adapter_lr)
        base_params = [p for p in base_module.parameters() if p.requires_grad]
        adapter_params = [p for p in adapter_module.parameters() if p.requires_grad]
        covered = {id(p) for p in base_module.parameters()}
        covered |= {id(p) for p in adapter_module.parameters()}
        other_params = [
            p for p in core.parameters() if p.requires_grad and id(p) not in covered
        ]
        split = []
        if base_params:
            split.append(('base', base_params, base_lr))
        if adapter_params:
            split.append(('adapter', adapter_params, adapter_lr))
        if other_params:
            split.append(('other', other_params, adapter_lr))
        return split or None
    # ===== TGCA_SPLIT_LR_V1_END =====

    def _select_criterion(self, loss_type=None):
        loss_type = loss_type or self.args.loss
        loss_type = loss_type.lower()
        if loss_type == 'mse':
            criterion = nn.MSELoss()
        elif loss_type == 'mae':
            criterion = nn.L1Loss()
        elif loss_type == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = loss_type
        # ===== TGCA_PS_CRITERION_V1 =====
        if (
            os.environ.get(
                "TGCA_PS_ENABLE",
                "0",
            ) == "1"
            and getattr(
                self.args,
                "model",
                "",
            ) in {
                "TGCA_Forecaster",
                "TGCA_CPTRD",
            }
        ):
            from utils.tgca_ps_loss import (
                TGCACompositePSLoss,
            )

            criterion = TGCACompositePSLoss(
                base_criterion=criterion,
                model=self.model,
                ps_lambda=float(
                    os.environ.get(
                        "TGCA_PS_LAMBDA",
                        "1.0",
                    )
                ),
                patch_max=int(
                    os.environ.get(
                        "TGCA_PS_PATCH_MAX",
                        "24",
                    )
                ),
                dynamic_weighting=(
                    os.environ.get(
                        "TGCA_PS_DYNAMIC",
                        "1",
                    ) == "1"
                ),
            )
        # ===== TGCA_PS_CRITERION_V1_END =====

        return criterion

    def forward_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if ('PEMS' in self.args.data or 'SRU' in self.args.data) and self.args.model not in ['TiDE']:
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        model_args = [batch_x, batch_x_mark, dec_inp, batch_y_mark]
        if self.args.model in MODEL_REQUIRES_CYCLE:
            model_args.append(batch_cycle)
        if self.args.output_attention:
            outputs, attn = self.model(*model_args)
        else:
            outputs, attn = self.model(*model_args), None

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.pred_len:, f_dim:]
        return outputs, batch_y, attn

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
