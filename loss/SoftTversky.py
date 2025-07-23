import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union


class TverskyLoss(nn.Module):
    eps = 1e-6
    def __init__(
            self,
            class_weight: Optional[Union[torch.Tensor, int, float, list, tuple, np.ndarray]] = None,
            alpha: float = 0.5, # α = β = 0.5 时，Tversky指数退化为Dice系数
            beta: float = 0.5, # 当 α = β = 1 时，Tversky指数退化为IoU
            gamma: float = 1,
            ignore_index: int = None,
            reduction: str = 'mean',
            label_smoothing: float = 0.05,
            num_classes: int = 2,
            dynamic_weights: bool = True,
            channel_first: bool = True,
    ):
        super(TverskyLoss, self).__init__()
        assert 0.0 <= label_smoothing < 1.0, "label_smoothing must be in [0, 1)"
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        assert 0.0 <= beta <= 1.0, "beta must be in [0, 1]"
        assert alpha + beta <= 2.0, 'alpha + beta must be less than or equal to'
        assert gamma >= 1.0, 'gamma must be greater than 1'

        self.class_weight = class_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = 255 if ignore_index is None else ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.num_classes = num_classes
        self.dynamic_weights = dynamic_weights
        self.channel_first = channel_first

    def _get_ignore_mask(self, targets) -> torch.Tensor:
        return targets != self.ignore_index

    def _transform_logits(self, logits: torch.Tensor):
        if not self.channel_first:
            logits = logits.permute(0, -1, *range(1, logits.ndim - 1))

        if logits.size(1) == 1:
            logits = torch.cat([-logits, logits], dim=1)
            self.num_classes = 2
        else:
            if logits.size(1) != self.num_classes:
                raise ValueError('特征的类别数和定义的类别数不匹配，请检查')

        return logits.contiguous()

    def _get_class_weight(self, targets: torch.Tensor):
        if self.class_weight is not None:
            class_weight = (
                torch.tensor(self.class_weight, dtype=torch.float32, device=targets.device)
                if not isinstance(self.class_weight, torch.Tensor)
                else self.class_weight.to(dtype=torch.float32, device=targets.device)
            )

            class_weight = class_weight.flatten()
            if class_weight.size(0) != self.num_classes:
                if class_weight.size(0) == 1 and self.num_classes == 2:
                    if class_weight[0] <= 0 or class_weight[0] >= 1:
                        raise ValueError('单一权重必须在0到1之间，请检查')
                    else:
                        class_weight = torch.cat([1.0 - class_weight, class_weight])
                else:
                    raise ValueError('权重数和类别不匹配，请检查')
            else:
                class_weight /= class_weight.sum()
        else:
            if not self.dynamic_weights:
                class_weight = torch.ones(self.num_classes, dtype=torch.float32, device=targets.device)
            else:
                # https://scikit-learn.org.cn/view/800.html
                class_counts = torch.bincount(targets[targets != self.ignore_index], minlength=self.num_classes).detach()
                total_pixels = class_counts.sum()

                if 0 in class_counts:
                    class_weight = torch.zeros_like(class_counts, dtype=torch.float32)
                    for i in range(self.num_classes):
                        if class_counts[i] != 0:
                            class_weight[i] = total_pixels / (self.num_classes * class_counts[i])
                else:
                    class_weight = total_pixels / (self.num_classes * class_counts)

        return class_weight.reshape(1, -1)

    def _get_smooth_label(self, targets):

        targets_onehot = F.one_hot(targets, self.num_classes).float()
        if self.label_smoothing > 0:
            targets_onehot = torch.clamp(
                targets_onehot, min=self.label_smoothing / (self.num_classes - 1), max=1 - self.label_smoothing)

        return targets_onehot.permute(0, -1, *range(1, targets_onehot.ndim - 1)).contiguous()

    def _get_tversky_index(self, logits, targets, class_weights):
        dim = tuple(range(2, logits.ndim))

        # Shape: (B, C)
        tp = (logits * targets).sum(dim=dim) * class_weights
        fp = (logits * (1.0 - targets)).sum(dim=dim) * class_weights
        fn = ((1.0 - logits) * targets).sum(dim=dim) * class_weights

        return (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)

    def _compute_loss(self, logits, targets):

        mask = self._get_ignore_mask(targets)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        logits = self._transform_logits(logits)
        logits = F.log_softmax(logits, dim=1).exp() * mask.unsqueeze(1)

        class_weights = self._get_class_weight(targets)
        targets = self._get_smooth_label(targets * mask)

        tversky_index = self._get_tversky_index(logits, targets, class_weights)

        return torch.pow(1.0 - tversky_index, 1.0 / self.gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        assert (logits.dim() >= 2 and targets.dim() >= 1) and (logits.dim() > targets.dim())

        logits = logits.to(dtype=torch.float32)
        loss = self._compute_loss(logits, targets)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "batchmean":
            return loss.sum() / logits.size(0)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction}")


if __name__ == "__main__":
    a = torch.randn(8, 2, 256, 256)
    b = torch.randint(0, 2, (8, 256, 256))
    A = TverskyLoss()
    c = A(a, b)
    print(' ')