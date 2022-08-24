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

def Dice_soft_func(preds: torch.Tensor, target: torch.Tensor):
    inter=torch.tensor(0.)
    union=torch.tensor(0.)
    pred,targ = flatten_check(torch.sigmoid(preds), target)
    inter += (pred*targ).sum().item()
    union += (pred+targ).sum().item()
    return 2.0 * inter/union if union > 0 else 0

# dice with automatic threshold selection
def Dice_threshold_func(preds: torch.Tensor, target: torch.Tensor, ths=np.arange(0.1,0.9,0.05)):
    inter=torch.zeros(len(ths))
    union=torch.zeros(len(ths))

    pred,targ = flatten_check(torch.sigmoid(preds), target)
    for i,th in enumerate(ths):
        p = (pred > th).float()
        inter[i] += (p*targ).sum().item()
        union[i] += (p+targ).sum().item()
    dices = torch.where(union > 0.0, 2.0 * inter/union, torch.zeros_like(union))
    return dices.max()