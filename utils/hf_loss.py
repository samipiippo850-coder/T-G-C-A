import torch
import torch.nn as nn
import pywt


class WaveletHighFreqLoss(nn.Module):
    """
    对预测与真实序列做小波分解，只比较高频细节分量。
    输入:
        pred: [B, T, C]
        true: [B, T, C]
    输出:
        scalar loss
    """
    def __init__(self, wavelet='db1', level=3, loss_type='l1', target_dim=None, device='cpu'):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.loss_type = loss_type
        self.target_dim = target_dim
        self.device = device

    def _one_series_loss(self, p, t):
        coeff_p = pywt.wavedec(p, wavelet=self.wavelet, level=self.level)
        coeff_t = pywt.wavedec(t, wavelet=self.wavelet, level=self.level)

        # coeff[0] 为低频近似项，coeff[1:] 为高频细节项
        loss = 0.0
        count = 0
        for cp, ct in zip(coeff_p[1:], coeff_t[1:]):
            cp = torch.tensor(cp, dtype=torch.float32, device=self.device)
            ct = torch.tensor(ct, dtype=torch.float32, device=self.device)
            if self.loss_type == 'l2':
                loss = loss + torch.mean((cp - ct) ** 2)
            else:
                loss = loss + torch.mean(torch.abs(cp - ct))
            count += 1
        return loss / max(count, 1)

    def forward(self, pred, true):
        # pred,true: [B,T,C]
        B, T, C = pred.shape
        dims = range(C) if self.target_dim is None else [self.target_dim]

        total = 0.0
        count = 0
        pred_cpu = pred.detach().cpu().numpy()
        true_cpu = true.detach().cpu().numpy()

        for b in range(B):
            for c in dims:
                total = total + self._one_series_loss(pred_cpu[b, :, c], true_cpu[b, :, c])
                count += 1

        if count == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=pred.device)

        return total / count
