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

from master_thesis_benge.self_supervised_learning.model.training.sim_clr_model import SimCLR_pl

from master_thesis_benge.self_supervised_learning.config.config_self_supervised_learning_training import training_config_resnet, training_config_unet, get_data_set_files

from master_thesis_benge.self_supervised_learning.config.constants import (
    PIPELINES_CONFIG_KEY,
    TRAINING_CONFIG_KEY,
    SEED_KEY,
    DATASET_SIZE_KEY,
    GRADIENT_ACCUMULATION_STEPS_KEY,
    EPOCHS_KEY,
    CHECKPOINT_PATH_KEY,
    FEATURE_DIMENSION_KEY,
    RESUME_FROM_CHECKPOINT_KEY,
    SAVE_MODEL_KEY,
    TRAINING_RESNET_CONFIG_KEY,
    TRAINING_UNET_CONFIG_KEY,
    get_label_from_index
)

from ffcv.loader import Loader, OrderOption

def select_training_config(config_type):
    if config_type == TRAINING_RESNET_CONFIG_KEY:
        return training_config_resnet
    elif config_type == TRAINING_UNET_CONFIG_KEY:
        return training_config_unet
    else:
        raise ValueError("Invalid config type.")

def reproducibility(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    #torch.cuda.manual_seed(seed)

def training():

    # Initialize wandb
    wandb.init()
    training_config = select_training_config(wandb.config.training_config)
    wandb.config.update(training_config)
    run_name = '-'.join([get_label_from_index(modality) for modality in wandb.config.modalities])
    wandb.run.name = run_name

    # Print available gpus
    available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    print("available_gpus:", available_gpus)

    # Model path
    save_model_path = os.path.join(os.getcwd(), training_config[TRAINING_CONFIG_KEY][SAVE_MODEL_KEY],str(wandb.run.sweep_id))
    filename = '-'.join([get_label_from_index(modality) for modality in wandb.config.modalities])+'-'+str(wandb.config.dataset_size)
    resume_from_checkpoint = training_config[TRAINING_CONFIG_KEY][RESUME_FROM_CHECKPOINT_KEY]

    reproducibility(training_config[TRAINING_CONFIG_KEY][SEED_KEY])

    #training_config[TRAINING_CONFIG_KEY][DATASET_SIZE_KEY]
    dataloader_train = Loader(get_data_set_files(wandb.config.dataset_size)[0],
                    batch_size=wandb.config.batch_size,
                    order=OrderOption.RANDOM,
                    num_workers=4,
                    pipelines=training_config[PIPELINES_CONFIG_KEY]
                )

    # Create a dictionary that maps each modality to the number of input channels
    
    channel_modalities = {
        f"in_channels_{i+1}": int(str(np.shape(next(iter(dataloader_train))[modality])[1]))
        for i, modality in enumerate(
            wandb.config.modalities
        )
    }

    model = SimCLR_pl(training_config, feat_dim=training_config[TRAINING_CONFIG_KEY][FEATURE_DIMENSION_KEY], in_channels_1=channel_modalities["in_channels_1"], in_channels_2=channel_modalities["in_channels_2"])

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
        save_top_k=1,
        monitor="Contrastive loss_epoch",
        mode="min",
    )

    if resume_from_checkpoint:
        trainer = Trainer(
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            max_epochs=training_config[TRAINING_CONFIG_KEY][EPOCHS_KEY],
            resume_from_checkpoint=training_config[TRAINING_CONFIG_KEY][CHECKPOINT_PATH_KEY], # -> check in config for correct path of checkpoint (if this needs to be executed at some point)
        )
    else:
        trainer = Trainer(
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            max_epochs=training_config[TRAINING_CONFIG_KEY][EPOCHS_KEY],
        )

    trainer.fit(model, dataloader_train)