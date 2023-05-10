from torch.utils.data import DataLoader
import numpy as np
import wandb
import torch

from master_thesis_benge.supervised_baseline.config.config_runs.config_segmentation import (
    training_config,
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
    PIPELINES_CONFIG_KEY
)

from master_thesis_benge.supervised_baseline.training.train import (
    Train,
)

from ffcv.loader import Loader, OrderOption

from master_thesis_benge.supervised_baseline.model.unet import UNet


if __name__ == "__main__":

    model = UNet(4,11)
    model.load_state_dict(torch.load('/netscratch2/nkesseli/master-thesis-benge/src/master_thesis_benge/supervised_baseline/saved_models/wjhv2od7/model.pt'))
    model.eval()

    sample = np.load('/ds2/remote_sensing/ben-ge/ben-ge/sentinel-2/S2A_MSIL2A_20170613T101031_46_73/S2A_MSIL2A_20170613T101031_46_73_all_bands.npy')

    np.save('/netscratch2/nkesseli/master-thesis-benge/src/master_thesis_benge/supervised_baseline/saved_models/wjhv2od7/input.npy',sample)

    infrared_sample = sample[[7,3,2,1],:,:].astype('float32')

    infrared_sample = np.expand_dims(infrared_sample, axis=0)

    print(np.shape(infrared_sample))

    output = model(torch.from_numpy(infrared_sample))

    print(output)

    print(np.shape(output))

    input('')

    output = output.cpu().detach().numpy()

    np.save('/netscratch2/nkesseli/master-thesis-benge/src/master_thesis_benge/supervised_baseline/saved_models/wjhv2od7/output.npy',output)

    '''
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
        project="master-thesis-supervised-"+(training_config[TASK_CONFIG_KEY][TASK_KEY].name).lower(),
        entity="nicikess",
        config=training_config,
    )

    dataloader_train = Loader(training_config[TRAINING_CONFIG_KEY][DATALOADER_TRAIN_FILE_KEY], 
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

    data = iter(dataloader_train)
    first = next(data)
    for key in first:
        print(key)
        input("")

    run_description = input("Describe run: ")
    wandb.log({"Run description": run_description})

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
        # Input channels for modality 1
        in_channels_1=channel_modalities["in_channels_1"],
        # Input channels for modality 2
        #in_channels_2=channel_modalities["in_channels_2"],
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
    '''