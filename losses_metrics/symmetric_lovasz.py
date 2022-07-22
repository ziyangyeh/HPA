from typing import Optional
from torch.nn.modules.loss import _Loss
from segmentation_models_pytorch.losses import constants, lovasz

class SymmetricLovaszLoss(_Loss):
    def __init__(
        self,
        mode: str,
        per_image: bool = False,
        ignore_index: Optional[int] = None,
        from_logits: bool = True,
    ):
        assert mode in {constants.BINARY_MODE, constants.MULTILABEL_MODE, constants.MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, y_pred, y_true):

        if self.mode in {constants.BINARY_MODE, constants.MULTILABEL_MODE}:
            loss = 0.5*(lovasz._lovasz_hinge(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index) + lovasz._lovasz_hinge(-y_pred, 1.0-y_true, per_image=self.per_image, ignore=self.ignore_index))
        elif self.mode == constants.MULTICLASS_MODE:
            y_pred = y_pred.softmax(dim=1)
            loss = 0.5*(lovasz._lovasz_softmax(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index) + + lovasz._lovasz_hinge(-y_pred, 1.0-y_true, per_image=self.per_image, ignore=self.ignore_index))
        else:
            raise ValueError("Wrong mode {}.".format(self.mode))
        return loss