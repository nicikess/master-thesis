from torch.utils.data import DataLoader
import numpy as np
import wandb
import torch
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, IntDecoder

from datetime import datetime

now = datetime.now()

from _master_thesis_benge_supervised_learning.supervised_baseline.config.config_runs.config_classification import (
    training_config,
    dataset_train,
    dataset_validation
)

from _master_thesis_benge_supervised_learning.supervised_baseline.config.constants import (
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
    DATALOADER_FILE_KEY
)

from _master_thesis_benge_supervised_learning.supervised_baseline.training.train import (
    Train,
)

from remote_sensing_core.transforms.ffcv.min_max_scaler import MinMaxScaler
from remote_sensing_core.transforms.ffcv.clipping import Clipping

from _master_thesis_benge_supervised_learning.supervised_baseline.training.dataloader_utils import DataloaderUtils



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
        project="master-thesis-supervised-"+(training_config[TASK_CONFIG_KEY][TASK_KEY].name).lower(),
        entity="nicikess",
        config=training_config,
    )
    

    print('start loading...'+now.strftime("%H:%M:%S"))

    # 'sentinel_2': [NDArrayDecoder(), Clipping([0, 10_000]), ToTensor()],
    
    "Error message: TypeError: only integer tensors of a single element can be converted to an index"
    dataloader_train = Loader('/ds2/remote_sensing/ben-ge/ffcv/ben-ge-20-train.beton', 
                                batch_size=training_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
                                order=OrderOption.RANDOM,
                                num_workers=4,
                                pipelines= {
                                            'climate_zone': [FloatDecoder(), ToTensor(), ToDevice(device)],
                                            #'elevation_difference_label': [FloatDecoder(), ToTensor(), ToDevice(device)],
                                            'era_5': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                                            'esa_worldcover': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                                            'glo_30_dem': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                                            #'multiclass_numeric_label': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                                            #'multiclass_one_hot_label': [ToTensor(), ToDevice(device)],
                                            'season_s1': [FloatDecoder(), ToTensor(), ToDevice(device)],
                                            'season_s2': [FloatDecoder(), ToTensor(), ToDevice(device)],
                                            'sentinel_1': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                                            'sentinel_2': [NDArrayDecoder(), Clipping([0, 10_000]), ToTensor(), ToDevice(device)], 
                                }
                            )
    
    #DataloaderUtils = (training_config[TRAINING_CONFIG_KEY][DATALOADER_FILE_KEY], training_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY])

    #it = iter(dataloader_train)
    #first = next(it)
    #print(len(first))
    #assert len(first) == 12   
    #for key in first:
        #print(key)
        #input('key')
    
    print('finished loading...'+now.strftime("%H:%M:%S"))

    '''    
    # Define training dataloader
    dataloader_train = DataLoader(
        dataset_train,
        training_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
        shuffle=True,
        num_workers=4,
    )
    '''

    # Define validation dataloader
    '''
    dataloader_validation = DataLoader(
        dataset_validation,
        training_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
        shuffle=True,
        num_workers=4,
    )'''

    dataloader_validation = Loader('/netscratch2/nkesseli/master-thesis-benge/src/_master_thesis_benge_supervised_learning/scripts/data-split/yaml_ffcv_config/ben-ge-20-validation.beton', 
                            batch_size=training_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
                            order=OrderOption.RANDOM,
                            num_workers=4,
                            pipelines= {
                                        'climate_zone': [FloatDecoder(), ToTensor(), ToDevice(device)],
                                        #'elevation_difference_label': [FloatDecoder(), ToTensor(), ToDevice(device)],
                                        'era_5': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                                        'esa_worldcover': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                                        'glo_30_dem': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                                        #'multiclass_numeric_label': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                                        #'multiclass_one_hot_label': [ToTensor(), ToDevice(device)],
                                        'season_s1': [FloatDecoder(), ToTensor(), ToDevice(device)],
                                        'season_s2': [FloatDecoder(), ToTensor(), ToDevice(device)],
                                        'sentinel_1': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                                        'sentinel_2': [NDArrayDecoder(), Clipping([0, 10_000]), ToTensor(), ToDevice(device)], 
                            }
                        )

    run_description = input("Describe run: ")
    wandb.log({"Run description": run_description})

    # Create a dictionary that maps each modality to the number of input channels
    channel_modalities = {
        f"in_channels_{i+1}": int(str(np.shape(next(iter(dataloader_train))[modality])[1]))
        for i, modality in enumerate(
            training_config[TRAINING_CONFIG_KEY][MODALITIES_KEY][MODALITIES_KEY]
        )
    }

    # TODO: change in channels
    # Define model
    model = training_config[MODEL_CONFIG_KEY][MODEL_KEY](
        # Define multi modal model
        # Input channels for s1
        #in_channels_1=#channel_modalities["in_channels_1"],
        in_channels_1=4,
        # Input channels for s2
        number_of_classes=training_config[MODEL_CONFIG_KEY][NUMBER_OF_CLASSES_KEY],
    )

    wandb.log({"model details": model})
    wandb.log({"Notes": f'Modalities: {training_config[TRAINING_CONFIG_KEY][MODALITIES_KEY][MODALITIES_KEY]} with data set train size: {len(dataset_train)}'})
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
    ).train()
