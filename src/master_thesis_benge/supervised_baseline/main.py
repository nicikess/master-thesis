from torch.utils.data import DataLoader
import numpy as np
import wandb
import torch

from master_thesis_benge.supervised_baseline.config.config_runs.config_classification import (
    training_config,
    get_data_set_files
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
    TASK_KEY,
    DATALOADER_TRAIN_FILE_KEY,
    DATALOADER_VALIDATION_FILE_KEY,
    BANDS_KEY,
    PIPELINES_CONFIG_KEY,
    DATA_CONFIG_KEY,
    DATASET_SIZE_KEY
)

from master_thesis_benge.supervised_baseline.training.train import (
    Train,
)

from ffcv.loader import Loader, OrderOption

if __name__ == "__main__":

    environment = training_config[OTHER_CONFIG_KEY][ENVIRONMENT_KEY]

    sweep_configuration = {
        "method": 'grid',
        "name": 'sweepy',
        "parameters": {
            "seed": {'values': [42, 43, 44, 45, 46]},
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='reproducibility-experiments-dataset-size')

    def run_sweep(size: str):

        wandb.init()

        # Set device
        if environment == "local":
            device = torch.device('cpu')
            print("Running locally")
        else:
            device = torch.device('cuda')

        # Set seed
        torch.manual_seed(wandb.config.seed)
        np.random.seed(wandb.config.seed)

        wandb.login(key="9da448bfaa162b572403e1551114a17058f249d0")
        wandb.init(
            project="master-thesis-supervised-"+(training_config[TASK_CONFIG_KEY][TASK_KEY].name).lower(),
            entity="nicikess",
            config=training_config,
        )

        dataloader_train = Loader(get_data_set_files(size),
                                batch_size=training_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
                                order=OrderOption.RANDOM,
                                num_workers=4,
                                pipelines=training_config[PIPELINES_CONFIG_KEY]
                            )

        dataloader_validation = Loader(get_data_set_files(size),
                                batch_size=training_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
                                order=OrderOption.RANDOM,
                                num_workers=4,
                                pipelines=training_config[PIPELINES_CONFIG_KEY]
                            )

        # Create a dictionary that maps each modality to the number of input channels
        channel_modalities = {
            f"in_channels_{i+1}": int(str(np.shape(next(iter(dataloader_train))[modality])[1]))
            for i, modality in enumerate(
                training_config[TRAINING_CONFIG_KEY][MODALITIES_KEY][MODALITIES_KEY]
            )
        }

        # Define model
        model = training_config[MODEL_CONFIG_KEY][MODEL_KEY](
            # Define multi modal model
            # Input channels for s1
            in_channels_1=channel_modalities["in_channels_1"],
            #in_channels_1=4,
            # Input channels for s2
            number_of_classes=training_config[MODEL_CONFIG_KEY][NUMBER_OF_CLASSES_KEY],
        )

        wandb.log({"model details": model})
        wandb.log({"Notes": f'Modalities: {training_config[TRAINING_CONFIG_KEY][MODALITIES_KEY][MODALITIES_KEY]} with data set train size: {len(dataloader_train)}'})
        #wandb.config.update(training_config)

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

    for size in training_config[DATA_CONFIG_KEY][DATASET_SIZE_KEY]:
        print("Size: "+size)
        wandb.agent(sweep_id, function=run_sweep(size))
