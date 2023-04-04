import pandas as pd
import torch
from torch.utils.data import DataLoader
from master_thesis_benge_supervised_learning.classification_baseline.training.train import HyperParameter
from master_thesis_benge_supervised_learning.classification_baseline.training.train import Train
import numpy as np
import wandb

from  master_thesis_benge_supervised_learning.classification_baseline.config.constants import LocalFilesAndDirectoryReferences, TrainingParameters
from master_thesis_benge_supervised_learning.classification_baseline.dataset.ben_ge_s import BenGeS

if __name__ == "__main__":

    # Set environment to remote or local
    environment = "remote"

    if environment == "local":

        # Set references to files and directories
        esa_world_cover_data_train = pd.read_csv(LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_TRAIN)
        esa_world_cover_data_validation = pd.read_csv(LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_VALIDATION)
        sentinel_1_2_metadata = pd.read_csv(LocalFilesAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV)
        era5_data = pd.read_csv(LocalFilesAndDirectoryReferences.ERA5_CSV)

        root_dir_s1 = LocalFilesAndDirectoryReferences.SENTINEL_1_DIRECTORY
        root_dir_s2 = LocalFilesAndDirectoryReferences.SENTINEL_2_DIRECTORY
        root_dir_world_cover = LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_DIRECTORY

        # Set device
        device = torch.device("cpu")

    if environment == "remote":

        # Change import based on dataset size
        if TrainingParameters.DATASET_SIZE_SMALL:
            from master_thesis_benge_supervised_learning.classification_baseline.config.constants import \
                RemoteFilesAndDirectoryReferencesSmall as RemoteFilesAndDirectoryReferences
        else:
            from master_thesis_benge_supervised_learning.classification_baseline.config.constants import \
                RemoteFilesAndDirectoryReferencesLarge as RemoteFilesAndDirectoryReferences

        # Set references to files and directories
        esa_world_cover_data_train = pd.read_csv(RemoteFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_TRAIN)
        esa_world_cover_data_validation = pd.read_csv(RemoteFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_VALIDATION)
        sentinel_1_2_metadata = pd.read_csv(RemoteFilesAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV)
        era5_data = pd.read_csv(RemoteFilesAndDirectoryReferences.ERA5_CSV)

        root_dir_s1 = RemoteFilesAndDirectoryReferences.SENTINEL_1_DIRECTORY
        root_dir_s2 = RemoteFilesAndDirectoryReferences.SENTINEL_2_DIRECTORY
        root_dir_world_cover = RemoteFilesAndDirectoryReferences.ESA_WORLD_COVER_DIRECTORY

        device = torch.device("cuda")

    # Define configurations
    torch.manual_seed(TrainingParameters.SEED)
    np.random.seed(TrainingParameters.SEED)

    if environment == "remote":
        training_parameters = TrainingParameters()
        config = vars(training_parameters)
        print(config)
        #wandb.login(key='9da448bfaa162b572403e1551114a17058f249d0')
        #wandb.init(project="master-thesis", entity="nicikess", config=config)

    # Create dataset
    dataset_train = BenGeS(
        esa_world_cover_data=esa_world_cover_data_train,
        sentinel_1_2_metadata=sentinel_1_2_metadata,
        era5_data=era5_data,
        root_dir_s1=root_dir_s1,
        root_dir_s2=root_dir_s2,
        root_dir_world_cover=root_dir_world_cover,
        wandb=wandb,
        number_of_classes=TrainingParameters.NUMBER_OF_CLASSES,
        bands=TrainingParameters.BANDS,
        transform=TrainingParameters.TRANSFORMS,
        normalization_value=TrainingParameters.NORMALIZATION_VALUE
    )
    wandb.log({"Dataset size": len(dataset_train)})

    dataset_validation = BenGeS(
        esa_world_cover_data=esa_world_cover_data_validation,
        sentinel_1_2_metadata=sentinel_1_2_metadata,
        era5_data=era5_data,
        root_dir_s1=root_dir_s1,
        root_dir_s2=root_dir_s2,
        root_dir_world_cover=root_dir_world_cover,
        wandb=wandb,
        number_of_classes=TrainingParameters.NUMBER_OF_CLASSES,
        bands=TrainingParameters.BANDS,
        transform=TrainingParameters.TRANSFORMS,
        normalization_value=TrainingParameters.NORMALIZATION_VALUE
    )

    #eda = ExploratoryDataAnalysis(
        #dataset, esa_world_cover_data, era5_data, sentinel_1_2_metadata
    #)
    #eda.distribution_barchart(modality="s1_img")

    # Define training dataloader
    train_dl = DataLoader(dataset_train, TrainingParameters.BATCH_SIZE, shuffle=TrainingParameters.SHUFFLE_TRAINING_DATA)

    # Define validation dataloader
    validation_dl = DataLoader(dataset_validation, TrainingParameters.BATCH_SIZE, shuffle=TrainingParameters.SHUFFLE_VALIDATION_DATA)

    # Set hyper parameters
    hyper_parameter = HyperParameter(
        epochs=TrainingParameters.EPOCHS,
        batch_size=TrainingParameters.BATCH_SIZE,
        learning_rate=TrainingParameters.LEARNING_RATE,
        optimizer=TrainingParameters.OPTIMIZER,
        scheduler=TrainingParameters.SCHEDULER,
        loss=TrainingParameters.LOSS
    )

    # Define model
    if TrainingParameters.MULTI_MODAL:
        # Define multi modal model
        model = TrainingParameters.MODEL(
            # Input channels for s1
            in_channels_1=TrainingParameters.NUMBER_OF_INPUT_CHANNELS_S1,
            # Input channels for s2
            in_channels_2=TrainingParameters.NUMBER_OF_INPUT_CHANNELS_S2,
            number_of_classes=TrainingParameters.NUMBER_OF_CLASSES,
        )
    else:
        # Define single modal model (usually s2)
        model = TrainingParameters.MODEL(
            number_of_input_channels=TrainingParameters.NUMBER_OF_INPUT_CHANNELS_S2,
            number_of_classes=TrainingParameters.NUMBER_OF_CLASSES,
        ).model
    # wandb.log({"Model": model})

    # Run training routing
    train = Train(
        model,
        train_dl=train_dl,
        validation_dl=validation_dl,
        number_of_classes=TrainingParameters.NUMBER_OF_CLASSES,
        device=device,
        wandb=wandb,
        hyper_parameter=hyper_parameter,
        environment=environment,
        multi_modal=TrainingParameters.MULTI_MODAL,
    ).train()