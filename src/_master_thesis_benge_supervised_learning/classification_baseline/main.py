from torch.utils.data import DataLoader
from _master_thesis_benge_supervised_learning.classification_baseline.training.train import (
    Train,
)
import numpy as np
import wandb
import torch

from _master_thesis_benge_supervised_learning.classification_baseline.config.config_runs.config_classification import (
    training_config,
    modalities_config,
)

from _master_thesis_benge_supervised_learning.classification_baseline.config.constants import (
    OTHER_CONFIG_KEY,
    S1_IMG_KEY,
    MODEL_CONFIG_KEY,
    MODALITIES_CONFIG_KEY,
    DATA_CONFIG_KEY,
    ENVIRONMENT_KEY,
    TRAINING_CONFIG_KEY,
    SEED_KEY,
    BANDS_KEY,
    BATCH_SIZE_KEY,
    TRANSFORMS_KEY,
)

from remote_sensing_core.ben_ge.ben_ge_dataset import BenGe
from _master_thesis_benge_supervised_learning.classification_baseline.config.config_runs.config_files_and_directories import (
    LocalFilesAndDirectoryReferences,
)

if __name__ == "__main__":
    environment = training_config[OTHER_CONFIG_KEY][ENVIRONMENT_KEY]

    # Set device
    if environment == "local":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    dataset_train = BenGe(
        data_index_path=LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_TRAIN,
        sentinel_1_2_metadata_path=LocalFilesAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV,
        **modalities_config
    )

    dataset_validation = BenGe(
        data_index_path=LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_VALIDATION,
        sentinel_1_2_metadata_path=LocalFilesAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV,
        **modalities_config
    )

    # Set seed
    torch.manual_seed(training_config[TRAINING_CONFIG_KEY][SEED_KEY])
    np.random.seed(training_config[TRAINING_CONFIG_KEY][SEED_KEY])

    if environment == "remote":
        wandb.login(key="9da448bfaa162b572403e1551114a17058f249d0")
        wandb.init(
            project="master-thesis-experiments-test",
            entity="nicikess",
            config=training_config,
        )

    # wandb.log({"Dataset size": len(dataset_train)})
    # wandb.log({"Dataset modalities": str(dataset_train)})

    # Define training dataloader
    dataloader_train = DataLoader(
        dataset_train,
        training_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
        shuffle=True,
        num_workers=4,
    )

    # Define validation dataloader
    dataloader_validation = DataLoader(
        dataset_validation,
        training_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
        shuffle=True,
        num_workers=4,
    )

    run_description = input("Describe run: ")
    # wandb.log({"Run description": run_description})

    sample = dataset_train[0]

    # Define model
    model = training_config[MODEL_CONFIG_KEY][MODEL_KEY]()
        # Define multi modal model
        (
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
