from typing import Any, Callable, Dict, List, Union, Optional, Tuple

import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.deeplabv3.decoder import ASPP
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.unetplusplus.decoder import CenterBlock

def build_from_config(cfg):
    if cfg.architecture=='unetplusplus-with-aspp-fpn':
        return UnetPlusPlus_with_ASPP_FPN(cfg.backbone, segmentation_channels= cfg.segmentation_channels, atrous_rates=tuple(cfg.atrous_rates))
    else:
        return getattr(smp, cfg.architecture)(cfg.backbone)

from segmentation_models_pytorch.base import modules as md
from fastai.vision.all import PixelShuffle_ICNR
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

        self.shuf = PixelShuffle_ICNR(in_channels, in_channels*2//2)

    def forward(self, x, skip=None):
        # x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = self.shuf(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UnetPlusPlus_with_ASPP_FPN_Decoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        aspp_out_channels,
        segmentation_channels=32,
        atrous_rates=(6,12,18),
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        self.aspp = ASPP(encoder_channels[0], aspp_out_channels, atrous_rates)
        self.fpn = FPNDecoder(tuple(list((encoder_channels[0],)+decoder_channels[:-1])[::-1]), segmentation_channels=segmentation_channels, encoder_depth=n_blocks, merge_policy="cat")

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        features = (self.aspp(features[0]),)+features[1:]
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                    )
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
        fpn_input = [features[0]]+[dense_x[f"x_{0}_{i}"] for i in range(self.depth)]
        fpn_out = self.fpn(*fpn_input[::-1])
        fpn_out = F.interpolate(fpn_out,scale_factor=4,mode='bilinear')
        return torch.cat((dense_x[f"x_{0}_{self.depth}"], fpn_out), dim=1)

from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.encoders import get_encoder
class UnetPlusPlus_with_ASPP_FPN(SegmentationModel):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            segmentation_channels: int = 32,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
            atrous_rates: Tuple = (6, 12 ,18),
        ):
            super().__init__()

            self.encoder = get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )

            self.decoder = UnetPlusPlus_with_ASPP_FPN_Decoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                aspp_out_channels=self.encoder.out_channels[-1],
                atrous_rates=atrous_rates,
                segmentation_channels=segmentation_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type,
            )

            self.segmentation_head = SegmentationHead(
                in_channels=decoder_channels[-1]+segmentation_channels*4,
                out_channels=classes,
                activation=activation,
                kernel_size=3,
            )

            if aux_params is not None:
                self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
            else:
                self.classification_head = None

            self.name = "unetplusplus-with-aspp-fpn-{}".format(encoder_name)
            self.initialize()

from timm.optim import create_optimizer_v2
from pl_bolts.optimizers import lr_scheduler
from losses_metrics import SymmetricLovaszLoss, Dice_soft, Dice_threshold
class LitModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.batch_size = cfg.data.batch_size
        self.cfg = cfg
        self.cfg_optimizer = self.cfg.train.optimizer
        self.cfg_scheduler = self.cfg.train.scheduler
        self.cfg_scheduler.epochs = cfg.train.epochs
        self.learning_rate = self.cfg.train.optimizer.learning_rate
        self.weight_decay = self.cfg.train.optimizer.weight_decay

        self.save_hyperparameters()

        self.model = build_from_config(cfg.model)

        self.loss_fn = self._init_loss_fn()

        self.dice_soft, self.dice_th = self._init_metric_fn()

    # def _init_loss_fn(self):
    #     # TODO: try other losses
    #     return monai.losses.DiceLoss(sigmoid=True)
    
    def _init_loss_fn(self):
        return SymmetricLovaszLoss("binary")

    def _init_metric_fn(self):
        return Dice_soft(), Dice_threshold()

    def configure_optimizers(self):
        # Setup the optimizer
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.cfg_optimizer.learning_rate, weight_decay=self.cfg_optimizer.weight_decay)

        # Setup the scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.cfg_scheduler.step_size,
                                                    gamma=self.cfg_scheduler.gamma)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def predict_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        images, masks = batch["image"], batch["mask"]
        outputs = self(images)

        loss = self.loss_fn(outputs, masks)
        dice_soft = self.dice_soft(outputs, masks)
        dice_th = self.dice_th(outputs, masks)

        self.log(f"{step}_loss", loss, batch_size=self.batch_size)
        self.log(f"{step}_dice_soft", dice_soft, batch_size=self.batch_size)
        self.log(f"{step}_dice_th", dice_th, batch_size=self.batch_size)

        return loss

    @classmethod
    def load_eval_checkpoint(cls, checkpoint_path: str, device: str) -> nn.Module:
        module = cls.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
        module.eval()

        return module
