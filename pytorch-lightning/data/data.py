from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import albumentations as A
import pytorch_lightning as pl

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations.pytorch import ToTensorV2

class HuBMAPDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img = cv2.imread(self.dataframe["image"][index], cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.dataframe["mask"][index],cv2.IMREAD_GRAYSCALE)
        mask[mask!=0]=1

        if self.transform is not None:
            augmented = self.transform(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']
        else:
            raise NotImplemented

        return {"image":img, "mask":mask}

class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        spatial_size: int,
        val_fold: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters(ignore="data_frame")

        self.data_frame = data_frame

        self.train_transform, self.val_transform, self.test_transform = self._init_transforms()

    def _init_transforms(self) -> Tuple[Callable, Callable, Callable]:
        spatial_size = (self.hparams.spatial_size, self.hparams.spatial_size)
        train_transform = A.Compose([A.HorizontalFlip(),
                                     A.VerticalFlip(),
                                     A.RandomRotate90(),
                                     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, border_mode=cv2.BORDER_REFLECT),
                                     A.OneOf([A.OpticalDistortion(p=0.3),
                                              A.GridDistortion(p=.1),A.PiecewiseAffine(p=0.3)], p=0.3),
                                     A.OneOf([A.HueSaturationValue(10,15,10),
                                              A.CLAHE(clip_limit=2),
                                              A.RandomBrightnessContrast()], p=0.3),
                                     A.Resize(height=spatial_size[0],width=spatial_size[1]),
                                     ToTensorV2()])

        val_transform = A.Compose([A.Resize(height=spatial_size[0],width=spatial_size[1]),
                                   ToTensorV2()])

        test_transform = A.Compose([A.Resize(height=spatial_size[0],width=spatial_size[1]),
                                   ToTensorV2()])

        return train_transform, val_transform, test_transform

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            train_df = self.data_frame[self.data_frame.fold != self.hparams.val_fold].reset_index(drop=True)
            val_df = self.data_frame[self.data_frame.fold == self.hparams.val_fold].reset_index(drop=True)

            self.train_dataset = self._dataset(train_df, transform=self.train_transform)
            self.val_dataset = self._dataset(val_df, transform=self.val_transform)

        if stage == "test" or stage is None:
            val_df = self.data_frame[self.data_frame.fold == self.hparams.val_fold].reset_index(drop=True)
            self.test_dataset = self._dataset(val_df, transform=self.test_transform)

    def _dataset(self, df: pd.DataFrame, transform: Callable) -> Dataset:
        return HuBMAPDataset(dataframe=df, transform=transform)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True, val=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, train=False, val=True)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: Dataset, train: bool = False, val: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True if train and val else False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True if train and val else False,
        )
