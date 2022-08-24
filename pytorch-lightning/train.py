# %%
import sys, os
from pathlib import Path
ROOT_DIR = Path(os.path.abspath(os.path.join(os.getcwd(), "..")))
BASE_DIR = ROOT_DIR / "pytorch-lightning"
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)

# %%
# import pytorch_lightning.trainer.connectors.checkpoint_connector as module_to_edit
# !code {module_to_edit.__file__}
# https://github.com/pytorch/pytorch/issues/80809#issuecomment-1173949444

# %%
from typing import Any, Callable, Dict, Optional, Tuple

import timm
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import segmentation_models_pytorch as smp


from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.notebook import tqdm

# %%
import pytorch_lightning as pl
from pytorch_lightning.strategies import *
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from data import LitDataModule
from model import LitModule

def train(
    cfg,
    fold: int,
    data_frame: pd.DataFrame,
) -> None:
    pl.seed_everything(cfg.seed)

    data_module = LitDataModule(
        val_fold=fold,
        data_frame=data_frame,
        spatial_size=cfg.data.spatial_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    data_module.setup()

    module = LitModule(cfg)

    model_checkpoint = ModelCheckpoint(cfg.train.checkpoint_dir,
                                        monitor="val_dice_th",
                                        mode="max",
                                        verbose=True,
                                        filename=f"{module.model.__class__.__name__}_{cfg.model.backbone}_{cfg.data.spatial_size}_{fold}",
                                        )

    trainer = pl.Trainer(
        default_root_dir=cfg.train.checkpoint_dir,
        accelerator=cfg.train.accelerator, 
        devices=[2, 3],
        strategy=DDPStrategy(find_unused_parameters=True) if cfg.train.strategy == "DDP" else cfg.train.strategy,
        benchmark=True,
        deterministic=False,
        callbacks=[model_checkpoint],
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        log_every_n_steps=5,
        logger=WandbLogger(name=f"{module.model.__class__.__name__}_{cfg.model.backbone}_{cfg.data.spatial_size}_{fold}", project=cfg.logger.wandb.project) if cfg.logger.wandb.use == True else False,
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.data.accumulate_grad_batches,
    )

    trainer.tune(module, datamodule=data_module)

    trainer.fit(module, datamodule=data_module, ckpt_path=os.path.join(os.getcwd(), cfg.train.checkpoint_dir, f"{module.model.__class__.__name__}_{cfg.model.backbone}_{cfg.data.spatial_size}_{fold}"+".ckpt") if os.path.exists(os.path.join(os.getcwd(), cfg.train.checkpoint_dir, f"{module.model.__class__.__name__}_{cfg.model.backbone}_{cfg.data.spatial_size}_{fold}"+".ckpt")) else None)
    
    return trainer

# %%
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "working"
CONFIG_DIR = BASE_DIR / "config"

COMPETITION_DATA_DIR = INPUT_DIR / "hubmap-organ-segmentation"
CROPPED_DATA_DIR = INPUT_DIR / "mmsegmentation512x512"

CONFIG_YAML_PATH = CONFIG_DIR / "default.yaml"

# %%
import glob
from utils import EasyConfig

cfg = EasyConfig()
cfg.load(CONFIG_YAML_PATH)

cfg_train = cfg.train
cfg_data = cfg.data
cfg_model = cfg.model

file_list = np.unique([os.path.basename(i).split(".")[0] for i in glob.glob(str(CROPPED_DATA_DIR)+"/*/*")])
kf = KFold(cfg_data.n_split, shuffle=True, random_state=cfg.seed)

df = pd.DataFrame()
df["image"] = glob.glob(str(CROPPED_DATA_DIR)+"/train/*")
df["mask"] = glob.glob(str(CROPPED_DATA_DIR)+"/masks/*")
for fold, (_, val_idx) in enumerate(kf.split(file_list)):
        df.loc[val_idx, "fold"] = fold
df.to_csv(os.path.join(str(CROPPED_DATA_DIR), "train.csv"), index=False)

# %%
import wandb
import gc
for fold in range(cfg_data.n_split):
    trainer = train(cfg, fold, df)
    wandb.finish()
    del trainer
    gc.collect()


