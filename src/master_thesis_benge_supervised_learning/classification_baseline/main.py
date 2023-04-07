import pandas as pd
from torch.utils.data import DataLoader
from master_thesis_benge_supervised_learning.classification_baseline.training.train import Train
import numpy as np
import wandb

from  master_thesis_benge_supervised_learning.classification_baseline.config.constants import LocalFilesAndDirectoryReferences
from master_thesis_benge_supervised_learning.classification_baseline.dataset.ben_ge_s import BenGeS
from master_thesis_benge_supervised_learning.classification_baseline.config.config import *

if __name__ == "__main__":

    environment = config["other"][environment_key]

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
        if config.get(data_set_size_small_key):
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
    torch.manual_seed(config["training"][seed_key])
    np.random.seed(config["training"][seed_key])

    if environment == "remote":
        wandb.login(key='9da448bfaa162b572403e1551114a17058f249d0')
        wandb.init(project="master-thesis-experiments", entity="nicikess", config=config)

    #wandb.log({"Dataset size": len(dataset_train)})

    #eda = ExploratoryDataAnalysis(
        #dataset, esa_world_cover_data, era5_data, sentinel_1_2_metadata
    #)
    #eda.distribution_barchart(modality="s1_img")

    # Define training dataloader
    train_dl = DataLoader(dataset_train, config['training'][batch_size_key], shuffle=config['data'][shuffle_training_data_key])

    # Define validation dataloader
    validation_dl = DataLoader(dataset_validation, config['training'][batch_size_key], shuffle=config['data'][shuffle_validation_data_key])

    #run_description = input("Describe run: ")
    #wandb.log({"Run description": run_description})

    # Define model
    if config['model'][multi_modal_key]:
        # Define multi modal model
        model = config['model'][model_key](
            # Input channels for s1
            in_channels_1=config['model'][number_of_input_channels_s1_key],
            # Input channels for s2
            in_channels_2=config['model'][number_of_input_channels_s2_key],
            number_of_classes=config['model'][number_of_classes_key],
            wandb=wandb
        ).model
    else:
        # Define single modal model (usually s2)
        model = config['model'][model_key](
            number_of_input_channels=config['model'][number_of_input_channels_s2_key],
            number_of_classes=config['model'][number_of_classes_key],
            wandb=wandb
        ).model
    #wandb.log({"model details": model})
    #wandb.config.update(config)

    # Run training routing
    train = Train(
        model,
        train_dl=train_dl,
        validation_dl=validation_dl,
        wandb=wandb,
        device=device,
    ).train()