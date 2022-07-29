from typing import Optional
import torch
import numpy as np
from fastai.vision.all import flatten_check
from torchmetrics import Metric

class Dice_soft(Metric):

    is_differentiable: Optional[bool] = None
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis 
        self.add_state("inter", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        pred,targ = flatten_check(torch.sigmoid(preds), target)

        self.inter += (pred*targ).sum().item()
        self.union += (pred+targ).sum().item()

    def compute(self):
        return 2.0 * self.inter/self.union if self.union > 0 else None

# dice with automatic threshold selection
class Dice_threshold(Metric):

    is_differentiable: Optional[bool] = None
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, ths=np.arange(0.1,0.9,0.05), axis=1):
        super().__init__() 
        self.axis = axis
        self.ths = ths
        self.add_state("inter", default=torch.zeros(len(self.ths)), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(len(self.ths)), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        pred,targ = flatten_check(torch.sigmoid(preds), target)
        for i,th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p*targ).sum().item()
            self.union[i] += (p+targ).sum().item()

    def compute(self):
        dices = torch.where(self.union > 0.0, 
                2.0*self.inter/self.union, torch.zeros_like(self.union))
        return dices.max()