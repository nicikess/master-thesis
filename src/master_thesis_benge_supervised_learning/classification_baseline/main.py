import pandas as pd
import torch
from torch.utils.data import DataLoader
from ben_ge_s import BenGeS
from train import HyperParameter
from train import Train
import numpy as np
import wandb
import torch.utils.data as data

from master_thesis_benge_supervised_learning.constants import (
    LocalFilesAndDirectoryReferences,
    TrainingParameters,
)

if __name__ == "__main__":

    # Set environment to remote or local
    environment = "local"

    if environment == "local":

        # Set references to files and directories
        sentinel_1_2_metadata = pd.read_csv(LocalFilesAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV.value)
        esa_world_cover_data = pd.read_csv(LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV.value)
        era5_data = pd.read_csv(LocalFilesAndDirectoryReferences.ERA5_CSV.value)

        root_dir_s1 = LocalFilesAndDirectoryReferences.SENTINEL_1_DIRECTORY.value
        root_dir_s2 = LocalFilesAndDirectoryReferences.SENTINEL_2_DIRECTORY.value
        root_dir_world_cover = LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_DIRECTORY.value

        # Set device
        device = torch.device("cpu")

    # TODO: Set references to files and directories
    if environment == "remote":
        data_index = pd.read_csv(
            "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy/ben-ge-s_esaworldcover.csv"
        )
        root_dir = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy/"
        device = torch.device("cuda")


    # Define configurations
    torch.manual_seed(TrainingParameters.SEED.value)
    np.random.seed(TrainingParameters.SEED.value)

    # if environment == "remote":
    # config = {i.name: i.value for i in TrainingParameters}
    # wandb.login(key='9da448bfaa162b572403e1551114a17058f249d0')
    # wandb.init(project="master-thesis", entity="nicikess", config=config)

    # Create dataset
    dataset = BenGeS(
        sentinel_1_2_metadata,
        esa_world_cover_data,
        era5_data,
        root_dir_s1,
        root_dir_s2,
        root_dir_world_cover,
        wandb=wandb,
        number_of_classes=TrainingParameters.NUMBER_OF_CLASSES.value,
        bands=TrainingParameters.BANDS.value,
        transform=TrainingParameters.TRANSFORMS.value,
        normalization_value=TrainingParameters.NORMALIZATION_VALUE.value
    )

    #eda = ExploratoryDataAnalysis(
        #dataset, esa_world_cover_data, era5_data, sentinel_1_2_metadata
    #)
    #eda.distribution_barchart(modality="s1_img")

    # Random split
    # TODO: Change to different train & validation csvs
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    train_ds, validation_ds = data.random_split(
        dataset, [train_set_size, valid_set_size],
    )
    # wandb.log({"Length dataset": len(dataset)})

    # Define training dataloader
    train_dl = DataLoader(train_ds, TrainingParameters.BATCH_SIZE.value, shuffle=True)

    # Define validation dataloader
    validation_dl = DataLoader(
        validation_ds, TrainingParameters.BATCH_SIZE.value, shuffle=True
    )

    # Set hyper parameters
    hyper_parameter = HyperParameter(
        epochs=TrainingParameters.EPOCHS.value,
        batch_size=TrainingParameters.BATCH_SIZE.value,
        learning_rate=TrainingParameters.LEARNING_RATE.value,
        optimizer=TrainingParameters.OPTIMIZER.value,
        scheduler=TrainingParameters.SCHEDULER.value,
        loss=TrainingParameters.LOSS.value
    )

    # Define model
    if TrainingParameters.MULTI_MODAL.value:
        # Define multi modal model
        model = TrainingParameters.MODEL.value(
            # Input channels for s1
            in_channels_1=TrainingParameters.NUMBER_OF_INPUT_CHANNELS_S1.value,
            # Input channels for s2
            in_channels_2=TrainingParameters.NUMBER_OF_INPUT_CHANNELS_S2.value,
            number_of_classes=TrainingParameters.NUMBER_OF_CLASSES.value,
        )
    else:
        # Define single modal model (usually s2)
        model = TrainingParameters.MODEL.value(
            number_of_input_channels=TrainingParameters.NUMBER_OF_INPUT_CHANNELS_S2.value,
            number_of_classes=TrainingParameters.NUMBER_OF_CLASSES.value,
        ).model
    # wandb.log({"Model": model})

    # Run training routing
    train = Train(
        model,
        train_dl=train_dl,
        validation_dl=validation_dl,
        number_of_classes=TrainingParameters.NUMBER_OF_CLASSES.value,
        device=device,
        wandb=wandb,
        hyper_parameter=hyper_parameter,
        environment=environment,
        multi_modal=TrainingParameters.MULTI_MODAL.value,
    ).train()
