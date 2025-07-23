import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from loss_functions.SoftCrossEntropy import CrossEntropyLoss
from loss_functions.SoftTversky import TverskyLoss
from typing import Optional, Union, Sequence


class ComboLoss(nn.Module):
    # aux_weight = (0.4, 0.3, 0.2, 0.1)
    def __init__(
            self,
            class_weight: Optional[Union[torch.Tensor, int, float, list, tuple, np.ndarray]] = None,
            ce_gamma: float = 0.,  # 当 γ = 0 时，不使用focal
            tversky_alpha: float = 0.5,  # α = β = 0.5 时，Tversky指数退化为Dice系数
            tversky_beta: float = 0.5,  # 当 α = β = 1 时，Tversky指数退化为IoU
            tversky_gamma: float = 1, # γ = 1 时，不使用focal
            ignore_index: int = None,
            label_smoothing: float = 0.05,
            num_classes: int = 2,
            dynamic_weights: bool = True,
            aux_weight: Sequence = None,
            channel_first: bool = True,
    ):
        super().__init__()
        self.class_weight, self.num_classes = class_weight, num_classes
        self.alpha, self.beta = tversky_alpha, tversky_beta
        self.ce_gamma, self.tversky_gamma = ce_gamma, tversky_gamma
        self.ignore_index, self.label_smoothing = ignore_index, label_smoothing
        self.aux_weight, self.dynamic_weights = aux_weight, dynamic_weights
        self.channel_first = channel_first

        self.ce_loss = CrossEntropyLoss(
            class_weight, ce_gamma, ignore_index, 'mean', label_smoothing, num_classes,
            dynamic_weights, channel_first
        )

        self.tversky_loss = TverskyLoss(
            class_weight, tversky_alpha, tversky_beta, tversky_gamma,
            ignore_index, 'mean', label_smoothing, num_classes,
            dynamic_weights, channel_first
        )

    def _get_aux_loss(self):
        if self.aux_weight:
            self.aux_ce_loss = CrossEntropyLoss(
                self.class_weight, ignore_index=self.ignore_index, label_smoothing=self.label_smoothing,
                num_classes=self.num_classes, dynamic_weights=self.dynamic_weights, channel_first=self.channel_first
            )
            self.aux_dice_loss = TverskyLoss(
                self.class_weight, ignore_index=self.ignore_index, label_smoothing=self.label_smoothing,
                num_classes=self.num_classes, dynamic_weights=self.dynamic_weights, channel_first=self.channel_first
            )

    def _main_loss(self, logits, targets):
        return 0.5 * self.ce_loss(logits, targets) + 0.5 * self.tversky_loss(logits, targets)

    def _aux_loss(self, logits, targets):
        return 0.5 * self.aux_ce_loss(logits, targets) + 0.5 * self.aux_dice_loss(logits, targets)

    def forward(self, logits, targets):
        loss = 0.
        if self.training and isinstance(logits, (list, tuple)) == 2:
            logit_main, logit_aux = logits

            assert len(logit_aux) == len(self.aux_weight), '辅助分支的权重与特征数不统一，请检查'
            for each_aux_weight, each_aux_logit in zip(self.aux_weight, logit_aux):
                each_aux_logit = F.interpolate(each_aux_logit,
                                               targets.size()[1:], mode='bilinear', align_corners=False)
                loss += each_aux_weight * self._aux_loss(each_aux_logit, targets)

            loss += self._main_loss(logit_main, targets)
        else:
            loss += self._main_loss(logits, targets)

        return loss


if __name__ == "__main__":
    _a = dict(
        class_weight= None,
    ce_gamma= 0.,  # 当 γ = 0 时，不使用focal
    tversky_alpha= 0.5,  # α = β = 0.5 时，Tversky指数退化为Dice系数
    tversky_beta= 0.5,  # 当 α = β = 1 时，Tversky指数退化为IoU
    tversky_gamma= 1,  # γ = 1 时，不使用focal
    ignore_index= None,
    label_smoothing= 0.05,
    num_classes= 2,
    dynamic_weights= True,
    aux_weight= None,
    channel_first=False,
    )
    a = torch.randn(8, 2, 256, 256)
    b = torch.randint(0, 2, (8, 256, 256))
    A = ComboLoss(**_a, channel_first=False)
    c = A(a, b)
    print(' ')
