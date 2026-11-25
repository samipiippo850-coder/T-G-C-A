import torch

def build_event_weight_torch(
    y_true,
    peak_alpha=1.2,
    valley_alpha=0.8,
    turning_alpha=1.5,
    eps=1e-6
):
    """
    y_true: [B, T, C] or [B, T]
    return: same shape weight
    """
    if y_true.dim() == 2:
        y = y_true.unsqueeze(-1)
    else:
        y = y_true

    mean = y.mean(dim=1, keepdim=True)
    std = y.std(dim=1, keepdim=True, unbiased=False) + eps
    z = (y - mean) / std

    peak_mask = (z > 1.0).float()
    valley_mask = (z < -1.0).float()

    dy = y[:, 1:, :] - y[:, :-1, :]
    sign = torch.sign(dy)
    turning = torch.zeros_like(y)
    turning[:, 1:-1, :] = (sign[:, 1:, :] * sign[:, :-1, :] < 0).float()

    weight = 1.0 + peak_alpha * peak_mask + valley_alpha * valley_mask + turning_alpha * turning

    if y_true.dim() == 2:
        weight = weight.squeeze(-1)

    return weight
