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
    MODEL_CONFIG_KEY,
    MODEL_KEY,
    ENVIRONMENT_KEY,
    TRAINING_CONFIG_KEY,
    SEED_KEY,
    BATCH_SIZE_KEY,
    NUMBER_OF_CLASSES_KEY,
    MULTICLASS_LABEL_KEY,
    WORLD_COVER_MODALITY_KEY
)

from remote_sensing_core.ben_ge.ben_ge_dataset import BenGe
from _master_thesis_benge_supervised_learning.classification_baseline.config.config_runs.config_files_and_directories import RemoteFilesAndDirectoryReferences as FileAndDirectoryReferences

if __name__ == "__main__":
    environment = training_config[OTHER_CONFIG_KEY][ENVIRONMENT_KEY]

    # Set device
    if environment == "local":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    dataset_train = BenGe(
        data_index_path=FileAndDirectoryReferences.ESA_WORLD_COVER_CSV_TRAIN,
        sentinel_1_2_metadata_path=FileAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV,
        **modalities_config
    )

    dataset_validation = BenGe(
        data_index_path=FileAndDirectoryReferences.ESA_WORLD_COVER_CSV_VALIDATION,
        sentinel_1_2_metadata_path=FileAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV,
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
    wandb.log({"Run description": run_description})
    
    modalities = dataset_train._get_modalities()

    # World cover modalities, since its only used to get the label
    modalities.remove("esa_worldcover")

    # Create a dictionary that maps each modality to the number of input channels
    channel_modalities = {f"in_channels_{i+1}": int(str(np.shape(dataset_train[0][modality])[0]))
                        for i, modality in enumerate(modalities)}

    # Define model
    model = training_config[MODEL_CONFIG_KEY][MODEL_KEY](
        # Define multi modal model
            # Input channels for s1
            in_channels_1=channel_modalities["in_channels_1"],
            # Input channels for s2
            #in_channels_2=channel_modalities["in_channels_2"],
            #in_channels_3=channel_modalities["in_channels_3"],
            number_of_classes=training_config[MODEL_CONFIG_KEY][NUMBER_OF_CLASSES_KEY],
        )
    wandb.log({"model details": model})
    wandb.log({"Notes": f'Modalities: {modalities} with data set train size: {len(dataset_train)}'})
    wandb.config.update(training_config)


    # Run training routing
    train = Train(
        model,
        train_dl=dataloader_train,
        validation_dl=dataloader_validation,
        wandb=wandb,
        device=device,
        config=training_config
    ).train()
