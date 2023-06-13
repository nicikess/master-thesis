import wandb
import torch
import os
import numpy as np

from torchvision.models import resnet18
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

from master_thesis_benge.self_supervised_learning.config.config_ssl_test import (
    training_config,
    get_data_set_files
)

from master_thesis_benge.supervised_baseline.config.constants import (
    OTHER_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_KEY,
    ENVIRONMENT_KEY,
    TRAINING_CONFIG_KEY,
    SEED_KEY,
    BATCH_SIZE_KEY,
    NUMBER_OF_CLASSES_KEY,
    MODALITIES_KEY,
    METRICS_KEY,
    METRICS_CONFIG_KEY,
    TASK_CONFIG_KEY,
    TASK_KEY,
    DATALOADER_TRAIN_FILE_KEY,
    DATALOADER_VALIDATION_FILE_KEY,
    BANDS_KEY,
    PIPELINES_CONFIG_KEY,
    DATA_CONFIG_KEY,
    DATASET_SIZE_KEY,
    SENTINEL_1_INDEX_KEY,
    CLIMATE_ZONE_INDEX_KEY,
    ERA_5_INDEX_KEY,
    SEASON_S2_INDEX_KEY,
    GLO_30_DEM_INDEX_KEY,
    SENTINEL_2_INDEX_KEY,
    MODALITIES_LABEL_KEY,
    ESA_WORLD_COVER_INDEX_KEY,
    get_label_from_index,
)

from ffcv.loader import Loader, OrderOption

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

    # Create a dictionary that maps each modality to the number of input channels
    channel_modalities = {
        f"in_channels_{i+1}": int(str(np.shape(next(iter(dataloader_train))[modality])[1]))
        for i, modality in enumerate(
            wandb.config.modalities
        )
    }

    model = SimCLR_pl(train_config, feat_dim=512, in_channels_1=channel_modalities["in_channels_1"], in_channels_2=channel_modalities["in_channels_2"])

    dataloader_train = Loader(get_data_set_files(wandb.config.dataset_size)[0],
                        batch_size=training_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
                        order=OrderOption.RANDOM,
                        num_workers=4,
                        pipelines=training_config[PIPELINES_CONFIG_KEY]
                    )
    '''   
    itera = iter(dataloader_train)
    first = next(itera)
    for data in first:
        print(data)
        print(data.shape)
        input("test")
    '''

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

    print("Running on CPU: Change train initializer in train module cpu -> gpu ")

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

    trainer.fit(model, dataloader_train)
    trainer.save(save_name)
