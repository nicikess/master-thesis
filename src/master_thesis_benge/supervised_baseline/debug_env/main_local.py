from torch.utils.data import DataLoader
import numpy as np
import wandb
import torch

from master_thesis_benge.supervised_baseline.debug_env.config_classification_local import (
    training_config,
    dataset_train,
    dataset_validation
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
    TASK_KEY
)

from master_thesis_benge.supervised_baseline.training.train import (
    Train,
)

if __name__ == "__main__":
    environment = training_config[OTHER_CONFIG_KEY][ENVIRONMENT_KEY]

    # Set device
    if environment == "local":
        device = torch.device('cpu')
        print("Running locally")
    else:
        device = torch.device('cuda')

    # Set seed
    torch.manual_seed(training_config[TRAINING_CONFIG_KEY][SEED_KEY])
    np.random.seed(training_config[TRAINING_CONFIG_KEY][SEED_KEY])

    wandb.login(key="9da448bfaa162b572403e1551114a17058f249d0")
    wandb.init(
        project="master-thesis-supervised-" + (training_config[TASK_CONFIG_KEY][TASK_KEY].name).lower(),
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

    # Create a dictionary that maps each modality to the number of input channels
    channel_modalities = {
        f"in_channels_{i + 1}": int(str(np.shape(dataset_train[0][modality])[0]))
        for i, modality in enumerate(
            training_config[TRAINING_CONFIG_KEY][MODALITIES_KEY][MODALITIES_KEY]
        )
    }

    # Define model
    model = training_config[MODEL_CONFIG_KEY][MODEL_KEY](
        # Define multi modal model
        # Input channels for s1
        in_channels_1=channel_modalities["in_channels_1"],
        # Input channels for s2
        number_of_classes=training_config[MODEL_CONFIG_KEY][NUMBER_OF_CLASSES_KEY],
    )

    wandb.log({"model details": model})
    wandb.log({
                  "Notes": f'Modalities: {training_config[TRAINING_CONFIG_KEY][MODALITIES_KEY][MODALITIES_KEY]} with data set train size: {len(dataset_train)}'})
    wandb.config.update(training_config)



    # Run training routing
    train = Train(
        model,
        train_dl=dataloader_train,
        validation_dl=dataloader_validation,
        metrics=training_config[METRICS_CONFIG_KEY][METRICS_KEY],
        wandb=wandb,
        device=device,
        config=training_config,
        task=training_config[TASK_CONFIG_KEY][TASK_KEY]
    ).train()