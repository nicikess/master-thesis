import wandb
import torch
import os
import numpy as np

from torchvision.models import resnet18
from torchvision.datasets import STL10
from torch.multiprocessing import cpu_count
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from master_thesis_benge.self_supervised_learning.config.config import Hparams
from master_thesis_benge.self_supervised_learning.model.model import SimCLR_pl
from master_thesis_benge.self_supervised_learning.augmentation.augmentation import (
    Augment,
)


def get_str1_dataloader_ST110(batch_size, transform=None, split="unlabeled"):
    st110 = STL10("./", split=split, transform=transform, download=True)
    return DataLoader(
        dataset=st110, batch_size=batch_size, num_workers=cpu_count() // 2
    )


def reproducibility(config):
    SEED = int(config.seed)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if config.cuda:
        torch.cuda.manual_seed(SEED)


def train():
    wandb.init()
    available_gpus = len(
        [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    )
    save_model_path = os.path.join(os.getcwd(), "saved_models/")
    print("available_gpus:", available_gpus)
    filename = "SimCLR_ResNet18_adam_"
    resume_from_checkpoint = False
    train_config = Hparams()

    reproducibility(train_config)
    save_name = filename + ".ckpt"

    model = SimCLR_pl(train_config, model=resnet18(pretrained=False), feat_dim=512)

    transform = Augment(train_config.img_size)
    data_loader = get_str1_dataloader_ST110(wandb.config.batch_size, transform)

    accumulator = GradientAccumulationScheduler(
        scheduling={0: train_config.gradient_accumulation_steps}
    )
    checkpoint_callback = ModelCheckpoint(
        filename=filename,
        dirpath=save_model_path,
        save_last=True,
        save_top_k=2,
        monitor="Contrastive loss_epoch",
        mode="min",
    )

    if resume_from_checkpoint:
        trainer = Trainer(
            callbacks=[accumulator, checkpoint_callback],
            accelerator="cpu",
            max_epochs=train_config.epochs,
            resume_from_checkpoint=train_config.checkpoint_path,
        )
    else:
        trainer = Trainer(
            callbacks=[accumulator, checkpoint_callback],
            accelerator="cpu",
            max_epochs=train_config.epochs,
        )

    trainer.fit(model, data_loader)
