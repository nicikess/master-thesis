import pandas as pd
from torch.utils.data import DataLoader
from _master_thesis_benge_supervised_learning.classification_baseline.training.train import Train
import numpy as np
import wandb
import torch

from _master_thesis_benge_supervised_learning.classification_baseline.dataset.ben_ge_s import BenGeS
from _master_thesis_benge_supervised_learning.classification_baseline.config.run_configs.config_sentinel_1_sentinel_2_altitude import config
from _master_thesis_benge_supervised_learning.classification_baseline.config.constants import *


if __name__ == "__main__":

    environment = config[OTHER_CONFIG_KEY][ENVIRONMENT_KEY]

    if environment == "local":

        # Create dataset
        dataset_train = BenGeS(
            root_dir_s1=LocalFilesAndDirectoryReferences.SENTINEL_1_DIRECTORY,
            root_dir_s2=LocalFilesAndDirectoryReferences.SENTINEL_2_DIRECTORY,
            root_dir_world_cover=LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_DIRECTORY,
            root_dir_glo_30_dem=LocalFilesAndDirectoryReferences.GLO_30_DIRECTORY,
            era5_data_path=LocalFilesAndDirectoryReferences.ERA5_CSV,
            esa_world_cover_index_path=LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_TRAIN,
            sentinel_1_2_metadata_path=LocalFilesAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV,
            s2_bands=Bands.RGB,
        )

        # Create dataset
        dataset_validation = BenGeS(
            root_dir_s1=LocalFilesAndDirectoryReferences.SENTINEL_1_DIRECTORY,
            root_dir_s2=LocalFilesAndDirectoryReferences.SENTINEL_2_DIRECTORY,
            root_dir_world_cover=LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_DIRECTORY,
            root_dir_glo_30_dem=LocalFilesAndDirectoryReferences.GLO_30_DIRECTORY,
            era5_data_path=LocalFilesAndDirectoryReferences.ERA5_CSV,
            esa_world_cover_index_path=LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_VALIDATION,
            sentinel_1_2_metadata_path=LocalFilesAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV,
            s2_bands=Bands.RGB,
        )
        # Set device
        device = torch.device("cpu")

    if environment == "remote":

        # Change import based on dataset size
        if config[DATA_CONFIG_KEY][DATA_SET_SIZE_SMALL_KEY]:
            from master_thesis_benge_supervised_learning.classification_baseline.config.constants import \
                RemoteFilesAndDirectoryReferencesSmall as RemoteFilesAndDirectoryReferences
        else:
            from master_thesis_benge_supervised_learning.classification_baseline.config.constants import \
                RemoteFilesAndDirectoryReferencesLarge as RemoteFilesAndDirectoryReferences

        # Create dataset
        dataset_train = BenGeS(
            root_dir_s1=RemoteFilesAndDirectoryReferences.SENTINEL_1_DIRECTORY,
            root_dir_s2=RemoteFilesAndDirectoryReferences.SENTINEL_2_DIRECTORY,
            root_dir_world_cover=RemoteFilesAndDirectoryReferences.ESA_WORLD_COVER_DIRECTORY,
            root_dir_glo_30_dem=RemoteFilesAndDirectoryReferences.GLO_30_DIRECTORY,
            era5_data_path=RemoteFilesAndDirectoryReferences.ERA5_CSV,
            esa_world_cover_index_path=RemoteFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_TRAIN,
            sentinel_1_2_metadata_path=RemoteFilesAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV,
            s2_bands=config[DATA_CONFIG_KEY][BANDS_KEY],
            multiclass_label_threshold=config[DATA_CONFIG_KEY][LABEL_THRESHOLD_KEY]
        )

        # Create dataset
        dataset_validation = BenGeS(
            root_dir_s1=RemoteFilesAndDirectoryReferences.SENTINEL_1_DIRECTORY,
            root_dir_s2=RemoteFilesAndDirectoryReferences.SENTINEL_2_DIRECTORY,
            root_dir_world_cover=RemoteFilesAndDirectoryReferences.ESA_WORLD_COVER_DIRECTORY,
            root_dir_glo_30_dem=RemoteFilesAndDirectoryReferences.GLO_30_DIRECTORY,
            era5_data_path=RemoteFilesAndDirectoryReferences.ERA5_CSV,
            esa_world_cover_index_path=RemoteFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_VALIDATION,
            sentinel_1_2_metadata_path=RemoteFilesAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV,
            s2_bands=config[DATA_CONFIG_KEY][BANDS_KEY],
            multiclass_label_threshold=config[DATA_CONFIG_KEY][LABEL_THRESHOLD_KEY]
        )

        device = torch.device("cuda")

    # Define configurations
    torch.manual_seed(config[TRAINING_CONFIG_KEY][SEED_KEY])
    np.random.seed(config[TRAINING_CONFIG_KEY][SEED_KEY])

    if environment == "remote":
        wandb.login(key='9da448bfaa162b572403e1551114a17058f249d0')
        wandb.init(project="master-thesis-experiments", entity="nicikess", config=config)

    wandb.log({"Config reference: ": config[OTHER_CONFIG_KEY][CONFIG_NAME_KEY]})
    wandb.log({"Dataset size": len(dataset_train)})

    # Define training dataloader
    train_dl = DataLoader(dataset_train, config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY], shuffle=config[DATA_CONFIG_KEY][SHUFFLE_TRAINING_DATA_KEY], num_workers=4)

    # Define validation dataloader
    validation_dl = DataLoader(dataset_validation, config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY], shuffle=config[DATA_CONFIG_KEY][SHUFFLE_VALIDATION_DATA_KEY], num_workers=4)

    run_description = input("Describe run: ")
    wandb.log({"Run description": run_description})

    # Define model
    if config[MODEL_CONFIG_KEY][MULTI_MODAL_KEY]:
        # Define multi modal model
        model = config[MODEL_CONFIG_KEY][MODEL_KEY](
            # Input channels for s1
            in_channels_1=config[MODEL_CONFIG_KEY][NUMBER_OF_INPUT_CHANNELS_S1_KEY],
            # Input channels for s2
            in_channels_2=config[MODEL_CONFIG_KEY][NUMBER_OF_INPUT_CHANNELS_S2_KEY],
            in_channels_3=config[MODEL_CONFIG_KEY][NUMBER_OF_INPUT_CHANNELS_ALTITUDE_KEY],
            number_of_classes=config[MODEL_CONFIG_KEY][NUMBER_OF_CLASSES_KEY],
        )
    else:
        # Define single modal model (usually s2)
        model = config[MODEL_CONFIG_KEY][MODEL_KEY](
            number_of_input_channels=config[MODEL_CONFIG_KEY][NUMBER_OF_INPUT_CHANNELS_S2_KEY],
            number_of_classes=config[MODEL_CONFIG_KEY][NUMBER_OF_CLASSES_KEY]
        ).model
    wandb.log({"model details": model})
    wandb.config.update(config)

    # Run training routing
    train = Train(
        model,
        train_dl=train_dl,
        validation_dl=validation_dl,
        wandb=wandb,
        device=device,
        config=config
    ).train()