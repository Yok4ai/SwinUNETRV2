#architecture.py
import os
import time
import torch
import pytorch_lightning as pl
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import PersistentDataset, list_data_collate, decollate_batch, DataLoader, load_decathlon_datalist, CacheDataset
from monai.inferers import sliding_window_inference
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from monai.data import DataLoader, Dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.timer import Timer
from torch.cuda.amp import GradScaler
import wandb
from pytorch_lightning.loggers import WandbLogger


class LightweightSwinUNETR(pl.LightningModule):
    def __init__(self, train_loader, val_loader, max_epochs=100, val_interval=1, learning_rate=1e-4,
                 feature_size=48, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), patch_size=2,
                 weight_decay=1e-5, warmup_epochs=5, drop_rate=0.0, attn_drop_rate=0.0,
                 roi_size=(128, 128, 128), sw_batch_size=4, overlap=0.25):
        super().__init__()
        self.save_hyperparameters()
        # SwinUNETR-V2 Configuration
        self.model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,  # Enable SwinUNETR-V2!
            spatial_dims=3,
            depths=depths,
            num_heads=num_heads,
            norm_name="instance",
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=0.0,
            downsample="mergingv2"  # Use improved merging for V2
        )