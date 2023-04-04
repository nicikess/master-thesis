import pandas as pd
import torch
from torch.utils.data import DataLoader
from master_thesis_benge_supervised_learning.classification_baseline.training.train import HyperParameter
from master_thesis_benge_supervised_learning.classification_baseline.training.train import Train
import numpy as np
import wandb
import os
import yaml

from master_thesis_benge_supervised_learning.classification_baseline.config.constants import TrainModel, LocalFilesAndDirectoryReferences, RemoteFilesAndDirectoryReferencesLarge
from master_thesis_benge_supervised_learning.classification_baseline.dataset.ben_ge_s import BenGeS

if __name__ == "__main__":

    # Read config
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(root_dir, 'classification_baseline/config', 'test_config.yaml')
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    environment = config['environment']['value']

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
        if config['training']['dataset_size_small']:
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
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])

    if environment == "remote":
        wandb.login(key='9da448bfaa162b572403e1551114a17058f249d0')
        wandb.init(project="master-thesis", entity="nicikess", config=config['training'])

    # Create dataset
    dataset_train = BenGeS(
        esa_world_cover_data=esa_world_cover_data_train,
        sentinel_1_2_metadata=sentinel_1_2_metadata,
        era5_data=era5_data,
        root_dir_s1=root_dir_s1,
        root_dir_s2=root_dir_s2,
        root_dir_world_cover=root_dir_world_cover,
        wandb=wandb,
        number_of_classes=config['training']['number_of_classes'],
        bands=config['training']['bands'],
        transform=config['training']['transforms'],
        normalization_value=config['training']['normalization_value'],
        label_threshold=config['training']['label_threshold']
    )
    #wandb.log({"Dataset size": len(dataset_train)})

    dataset_validation = BenGeS(
        esa_world_cover_data=esa_world_cover_data_validation,
        sentinel_1_2_metadata=sentinel_1_2_metadata,
        era5_data=era5_data,
        root_dir_s1=root_dir_s1,
        root_dir_s2=root_dir_s2,
        root_dir_world_cover=root_dir_world_cover,
        wandb=wandb,
        number_of_classes=config['training']['number_of_classes'],
        bands=config['training']['bands'],
        transform=config['training']['transforms'],
        normalization_value=config['training']['normalization_value'],
        label_threshold=config['training']['label_threshold']
    )

    #eda = ExploratoryDataAnalysis(
        #dataset, esa_world_cover_data, era5_data, sentinel_1_2_metadata
    #)
    #eda.distribution_barchart(modality="s1_img")

    # Define training dataloader
    train_dl = DataLoader(dataset_train, config['training']['batch_size'], shuffle=config['training']['shuffle_training_data'])

    # Define validation dataloader
    validation_dl = DataLoader(dataset_validation, config['training']['batch_size'], shuffle=config['training']['shuffle_validation_data'])

    print(type(config['training']['learning_rate']))

    # Set hyper parameters
    hyper_parameter = HyperParameter(
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        optimizer=config['training']['optimizer'],
        scheduler=config['training']['scheduler'],
        loss=config['training']['loss'],
        scheduler_max_number_iterations=config['training']['scheduler_max_number_iterations'],
        scheduler_min_lr=config['training']['scheduler_min_lr'],
    )

    # Define model
    if config['training']['multi_modal']:
        # Define multi modal model
        model = config['training']['model'](
            # Input channels for s1
            in_channels_1=config['training']['number_of_input_channels_s1'],
            # Input channels for s2
            in_channels_2=config['training']['number_of_input_channels_s2'],
            number_of_classes=config['training']['number_of_classes'],
        )
    else:
        # Define single modal model (usually s2)
        model = TrainModel.MODEL(
            number_of_input_channels=config['training']['number_of_input_channels_s2'],
            number_of_classes=config['training']['number_of_classes']
        ).model
    # wandb.log({"Model": model})

    # Run training routing
    train = Train(
        model,
        train_dl=train_dl,
        validation_dl=validation_dl,
        number_of_classes=config['training']['number_of_classes'],
        device=device,
        wandb=wandb,
        environment=environment,
        multi_modal=config['training']['multi_modal'],
        save_model=config['training']['save_model'],
        hyper_parameter=hyper_parameter,
    ).train()