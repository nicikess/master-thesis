from torch.utils.data import DataLoader
import numpy as np
import wandb
import torch

from master_thesis_benge.supervised_baseline.config.config_runs.config_classification_landuse import (
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
    DATASET_SIZE_KEY,
    SENTINEL_1_INDEX_KEY,
    CLIMATE_ZONE_INDEX_KEY,
    ERA_5_INDEX_KEY,
    SEASON_S2_INDEX_KEY,
    GLO_30_DEM_INDEX_KEY,
    SENTINEL_2_INDEX_KEY,
    MODALITIES_LABEL_KEY,
    ESA_WORLD_COVER_INDEX_KEY,
    get_label_from_index,
)

from master_thesis_benge.supervised_baseline.training.train import (
    Train,
)

from ffcv.loader import Loader, OrderOption

if __name__ == "__main__":

    environment = training_config[OTHER_CONFIG_KEY][ENVIRONMENT_KEY]

    sweep_configuration = {
        "method": 'grid',
        "name": 'one-modality-seed-42-43-44-45-46',
        "parameters": {
            "seed": {'values': [42, 43, 44, 45, 46]},
            #"learning_rate": {'values': [0.0001]},
            "dataset_size": {'values': ["20"]},
            "modalities": {'values':    [
                                            [SENTINEL_2_INDEX_KEY],
                                            [SENTINEL_1_INDEX_KEY],
                                            [ESA_WORLD_COVER_INDEX_KEY],
                                            [ERA_5_INDEX_KEY],
                                            [GLO_30_DEM_INDEX_KEY],
                                            [CLIMATE_ZONE_INDEX_KEY],
                                            [SEASON_S2_INDEX_KEY],
                                        ]
                           },
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='master-thesis-supervised-'+training_config[TASK_CONFIG_KEY][TASK_KEY].lower())

    def run_sweep():

        wandb.init(config=training_config)
        run_name = '-'.join([get_label_from_index(modality) for modality in wandb.config.modalities])
        wandb.run.name = run_name

        # Set device
        if environment == "local":
            device = torch.device('cpu')
            print("Running locally")
        else:
            device = torch.device('cuda')

        # Set seed
        torch.manual_seed(wandb.config.seed)
        np.random.seed(wandb.config.seed)

        dataloader_train = Loader(get_data_set_files(wandb.config.dataset_size)[0],
                                batch_size=training_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
                                order=OrderOption.RANDOM,
                                num_workers=4,
                                pipelines=training_config[PIPELINES_CONFIG_KEY]
                            )

        dataloader_validation = Loader(training_config[TRAINING_CONFIG_KEY][DATALOADER_VALIDATION_FILE_KEY],
                                batch_size=training_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
                                order=OrderOption.RANDOM,
                                num_workers=4,
                                pipelines=training_config[PIPELINES_CONFIG_KEY]
                            )
        
        '''
        itera = iter(dataloader_train)
        first = next(itera)
        for data in first:
            print(data)
            print(data.shape)
            input("test")
        '''
        
               
        # Create a dictionary that maps each modality to the number of input channels
        channel_modalities = {
            f"in_channels_{i+1}": int(str(np.shape(next(iter(dataloader_train))[modality])[1]))
            for i, modality in enumerate(
                wandb.config.modalities
            )
        }

        itera = iter(dataloader_train)
        first = next(itera)
        for data in first:
            print(data)
            input("test")

        # Define model
        model = training_config[MODEL_CONFIG_KEY][MODEL_KEY](
            # Define multi modal model
            # Input channels for s1
            in_channels_1=channel_modalities["in_channels_1"],
            #in_channels_1=4,
            # Input channels for s2
            #in_channels_2=channel_modalities["in_channels_2"],
            #in_channels_3=channel_modalities["in_channels_3"],
            number_of_classes=training_config[MODEL_CONFIG_KEY][NUMBER_OF_CLASSES_KEY],
        )

        wandb.log({"model details": model})
        wandb.log({"Notes": f'Modalities: {wandb.config.modalities} with data set train size: {len(dataloader_train)}'})

        # Run training routing
        train = Train(
            model,
            train_dl=dataloader_train,
            validation_dl=dataloader_validation,
            metrics=training_config[METRICS_CONFIG_KEY][METRICS_KEY],
            wandb=wandb,
            device=device,
            config=training_config,
            task=training_config[TASK_CONFIG_KEY][TASK_KEY],
            modalities=wandb.config.modalities
        ).train()

    wandb.agent(sweep_id, function=run_sweep)