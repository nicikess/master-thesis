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

from master_thesis_benge.self_supervised_learning.model.model import SimCLR_pl
from master_thesis_benge.self_supervised_learning.augmentation.augmentation import (
    Augment,
)

from master_thesis_benge.self_supervised_learning.config.config_self_supervised_learning import (
    training_config,
    get_data_set_files
)

from master_thesis_benge.self_supervised_learning.config.constants import (
    PIPELINES_CONFIG_KEY,
    PARAMETERS_CONFIG_KEY,
    SEED_KEY,
    DATASET_SIZE_KEY,
    GRADIENT_ACCUMULATION_STEPS_KEY,
    EPOCHS_KEY,
    CHECKPOINT_PATH_KEY
)

from ffcv.loader import Loader, OrderOption

def reproducibility(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    #torch.cuda.manual_seed(seed)

def train():

    # Initialize wandb
    wandb.init(config=training_config)

    # Print available gpus

    available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    print("available_gpus:", available_gpus)

    # Model path
    save_model_path = os.path.join(os.getcwd(), "saved_models/")
    filename = "SimCLR_ResNet18_adam_"
    resume_from_checkpoint = False

    reproducibility(training_config[PARAMETERS_CONFIG_KEY][SEED_KEY])
    save_name = filename + ".ckpt"

    dataloader_train = Loader(get_data_set_files(training_config[PARAMETERS_CONFIG_KEY][DATASET_SIZE_KEY])[0],
                    batch_size=wandb.config.batch_size,
                    order=OrderOption.RANDOM,
                    num_workers=4,
                    pipelines=training_config[PIPELINES_CONFIG_KEY]
                )

    # Create a dictionary that maps each modality to the number of input channels
    
    '''
    channel_modalities = {
        f"in_channels_{i+1}": int(str(np.shape(next(iter(dataloader_train))[modality])[1]))
        for i, modality in enumerate(
            wandb.config.modalities
        )
    }
    '''

    model = SimCLR_pl(training_config, feat_dim=512, in_channels_1=2, in_channels_2=4)

    '''
    itera = iter(dataloader_train)
    first = next(itera)
    for data in first:
        print(data)
        print(data.shape)
        input("test")
    '''

    '''
    accumulator = GradientAccumulationScheduler(
        scheduling={0: training_config[PARAMETERS_CONFIG_KEY][GRADIENT_ACCUMULATION_STEPS_KEY]}
    )
    '''

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
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            max_epochs=training_config[PARAMETERS_CONFIG_KEY][EPOCHS_KEY],
            resume_from_checkpoint=training_config[PARAMETERS_CONFIG_KEY][CHECKPOINT_PATH_KEY],
        )
    else:
        trainer = Trainer(
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            max_epochs=training_config[PARAMETERS_CONFIG_KEY][EPOCHS_KEY],
        )

    trainer.fit(model, dataloader_train)
    trainer.save(save_name)
