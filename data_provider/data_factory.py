import os
from data_provider.data_loader import Dataset_Custom, Dataset_Custom_CCA, Dataset_Custom_FA, Dataset_Custom_Fourier, \
    Dataset_Custom_ICA, Dataset_Custom_PCA, Dataset_Custom_RobustICA, Dataset_Custom_RobustPCA, Dataset_Custom_SVD, \
    Dataset_ETT_hour, Dataset_ETT_hour_CCA, Dataset_ETT_hour_Fourier, Dataset_ETT_hour_PCA, Dataset_ETT_hour_Trend, \
    Dataset_ETT_minute, Dataset_ETT_minute_CCA, Dataset_ETT_minute_Fourier, Dataset_ETT_minute_PCA, Dataset_M4, \
    Dataset_M4_CCA, Dataset_M4_PCA, Dataset_PEMS, Dataset_PEMS_CCA, Dataset_PEMS_PCA, Dataset_Solar, Dataset_SRU, \
    MSLSegLoader, PSMSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader


# ===== PMRT TRAIN AUGMENTATION START =====
import os as _pmrt_os
import random as _pmrt_random
import torch as _pmrt_torch

class PMRTTrainAugWrapper:
    """
    Train-only augmentation wrapper for forecasting.
    It preserves input-label consistency by augmenting x and y together.

    Supported:
      PMRT_AUG_ENABLE=1
      PMRT_AUG_MIXUP_P=0.2
      PMRT_AUG_FREQMASK_P=0.2
      PMRT_AUG_FREQMASK_RATIO=0.05
      PMRT_AUG_NOISE_STD=0.0
    """
    def __init__(self, base):
        self.base = base
        self.mixup_p = float(_pmrt_os.environ.get("PMRT_AUG_MIXUP_P", "0.0"))
        self.freqmask_p = float(_pmrt_os.environ.get("PMRT_AUG_FREQMASK_P", "0.0"))
        self.freqmask_ratio = float(_pmrt_os.environ.get("PMRT_AUG_FREQMASK_RATIO", "0.05"))
        self.noise_std = float(_pmrt_os.environ.get("PMRT_AUG_NOISE_STD", "0.0"))
        self.alpha = float(_pmrt_os.environ.get("PMRT_AUG_MIXUP_ALPHA", "0.2"))

    def __len__(self):
        return len(self.base)

    def _to_tensor(self, z):
        if isinstance(z, _pmrt_torch.Tensor):
            return z.float()
        return _pmrt_torch.as_tensor(z).float()

    def _freq_mask(self, z):
        # z: [T, C]
        if z.ndim != 2:
            return z
        if self.freqmask_p <= 0 or _pmrt_random.random() > self.freqmask_p:
            return z

        Z = _pmrt_torch.fft.rfft(z, dim=0)
        n_freq = Z.shape[0]
        if n_freq <= 4:
            return z

        k = max(1, int(n_freq * self.freqmask_ratio))
        # do not mask DC component
        idx = _pmrt_torch.randperm(n_freq - 1)[:k] + 1
        Z[idx, :] = 0
        out = _pmrt_torch.fft.irfft(Z, n=z.shape[0], dim=0)
        return out.to(z.dtype)

    def _augment_pair(self, x, y):
        # Apply same type of augmentation to x and y so forecasting relation is preserved.
        x = self._freq_mask(x)
        y = self._freq_mask(y)

        if self.noise_std > 0:
            x = x + self.noise_std * _pmrt_torch.randn_like(x)
            y = y + self.noise_std * _pmrt_torch.randn_like(y)

        return x, y

    def __getitem__(self, idx):
        item = self.base[idx]
        if len(item) < 4:
            return item

        x, y, x_mark, y_mark = item[:4]
        rest = item[4:]

        x = self._to_tensor(x)
        y = self._to_tensor(y)

        # frequency/noise augmentation
        x, y = self._augment_pair(x, y)

        # mixup on full forecasting pair
        if self.mixup_p > 0 and _pmrt_random.random() < self.mixup_p:
            j = _pmrt_random.randrange(len(self.base))
            item2 = self.base[j]
            x2, y2, x_mark2, y_mark2 = item2[:4]
            x2 = self._to_tensor(x2)
            y2 = self._to_tensor(y2)

            # Beta(alpha, alpha), implemented via torch distribution
            beta = _pmrt_torch.distributions.Beta(self.alpha, self.alpha)
            lam = float(beta.sample().item())

            x = lam * x + (1.0 - lam) * x2
            y = lam * y + (1.0 - lam) * y2

            # keep original time marks; values are mixed, timestamps remain current sample
            # This avoids fractional/invalid time features.

        return (x, y, x_mark, y_mark, *rest)
# ===== PMRT TRAIN AUGMENTATION END =====


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh1_Fourier': Dataset_ETT_hour_Fourier,
    'ETTh1_Trend': Dataset_ETT_hour_Trend,
    'ETTh1_PCA': Dataset_ETT_hour_PCA,
    'ETTh1_CCA': Dataset_ETT_hour_CCA,
    'ETTh2': Dataset_ETT_hour,
    'ETTh2_PCA': Dataset_ETT_hour_PCA,
    'ETTh2_CCA': Dataset_ETT_hour_CCA,
    'ETTm1': Dataset_ETT_minute,
    'ETTm1_Fourier': Dataset_ETT_minute_Fourier,
    'ETTm1_PCA': Dataset_ETT_minute_PCA,
    'ETTm1_CCA': Dataset_ETT_minute_CCA,
    'ETTm2': Dataset_ETT_minute,
    'ETTm2_PCA': Dataset_ETT_minute_PCA,
    'ETTm2_CCA': Dataset_ETT_minute_CCA,
    'custom': Dataset_Custom,
    'custom_Fourier': Dataset_Custom_Fourier,
    'custom_FA': Dataset_Custom_FA,
    'custom_PCA': Dataset_Custom_PCA,
    'custom_RobustPCA': Dataset_Custom_RobustPCA,
    'custom_SVD': Dataset_Custom_SVD,
    'custom_ICA': Dataset_Custom_ICA,
    'custom_RobustICA': Dataset_Custom_RobustICA,
    'custom_CCA': Dataset_Custom_CCA,
    'PEMS': Dataset_PEMS,
    'PEMS_PCA': Dataset_PEMS_PCA,
    'PEMS_CCA': Dataset_PEMS_CCA,
    'Solar': Dataset_Solar,
    'm4': Dataset_M4,
    'm4_PCA': Dataset_M4_PCA,
    'm4_CCA': Dataset_M4_CCA,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'SRU': Dataset_SRU,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = args.test_batch_size  # bsz for test
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        # JOINT_ORDERED_TRAIN: preserve temporal adjacency for regime-gradient training.
        if (
            flag == "train"
            and os.environ.get(
                "JOINT_ORDERED_TRAIN", "0"
            ) == "1"
        ):
            shuffle_flag = False
            drop_last = False
            print(
                "[JOINT_ORDERED_TRAIN] "
                "shuffle=False drop_last=False",
                flush=True,
            )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=(
            False
            if (
                flag == "val"
                and os.getenv(
                    "CPTRD_FORCE_VAL_SEQUENTIAL",
                    "0",
                ) == "1"
            )
            else shuffle_flag
        ),
            num_workers=args.num_workers,
            drop_last=(
            False
            if (
                flag == "val"
                and os.getenv(
                    "CPTRD_FORCE_VAL_SEQUENTIAL",
                    "0",
                ) == "1"
            )
            else drop_last
        )
        )
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=(
            False
            if (
                flag == "val"
                and os.getenv(
                    "CPTRD_FORCE_VAL_SEQUENTIAL",
                    "0",
                ) == "1"
            )
            else shuffle_flag
        ),
            num_workers=args.num_workers,
            drop_last=(
            False
            if (
                flag == "val"
                and os.getenv(
                    "CPTRD_FORCE_VAL_SEQUENTIAL",
                    "0",
                ) == "1"
            )
            else drop_last
        ),
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            add_noise=args.add_noise,
            noise_amp=args.noise_amp,
            noise_freq_percentage=args.noise_freq_percentage,
            noise_seed=args.noise_seed,
            noise_type=args.noise_type,
            data_percentage=args.data_percentage,
            rank_ratio=args.rank_ratio,
            pca_dim=args.pca_dim,
            reinit=args.reinit,
            shift=args.shift,
            num_freqs=args.num_freqs,
            speedup_sklearn=args.speedup_sklearn,
            align_type=args.align_type,
            cycle=args.cycle,
            exclude_target_input=args.exclude_target_input,
            load_from_disk=args.load_from_disk,
        )
        print(flag, len(data_set))
        if flag == "train" and _pmrt_os.environ.get("PMRT_AUG_ENABLE", "0") == "1":
            data_set = PMRTTrainAugWrapper(data_set)
            print("[PMRT_AUG] train wrapper enabled:",
                  "mixup_p=", _pmrt_os.environ.get("PMRT_AUG_MIXUP_P", "0.0"),
                  "alpha=", _pmrt_os.environ.get("PMRT_AUG_MIXUP_ALPHA", "0.2"),
                  "freqmask_p=", _pmrt_os.environ.get("PMRT_AUG_FREQMASK_P", "0.0"),
                  "freqmask_ratio=", _pmrt_os.environ.get("PMRT_AUG_FREQMASK_RATIO", "0.05"),
                  "noise_std=", _pmrt_os.environ.get("PMRT_AUG_NOISE_STD", "0.0"))
            print("[PMRT_AUG] wrapped train len:", len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=(
            False
            if (
                flag == "val"
                and os.getenv(
                    "CPTRD_FORCE_VAL_SEQUENTIAL",
                    "0",
                ) == "1"
            )
            else shuffle_flag
        ),
            num_workers=args.num_workers,
            drop_last=(
            False
            if (
                flag == "val"
                and os.getenv(
                    "CPTRD_FORCE_VAL_SEQUENTIAL",
                    "0",
                ) == "1"
            )
            else drop_last
        )
        )
        return data_set, data_loader
