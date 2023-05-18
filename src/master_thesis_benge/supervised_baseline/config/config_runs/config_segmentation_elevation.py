import torch

# Import transforms
from master_thesis_benge.supervised_baseline.training.transforms import (
    Transforms,
)

# Import models
from master_thesis_benge.supervised_baseline.model.unet import (
    UNet
)

from master_thesis_benge.supervised_baseline.model.dual_unet import (
    DualUNet,
)

# Import constants
from master_thesis_benge.supervised_baseline.config.constants import (
    Task,
    WEIGHTS_KEY,
    NUMBER_OF_CLASSES_KEY,
    EPOCHS_KEY,
    LEARNING_RATE_KEY,
    BATCH_SIZE_KEY,
    OPTIMIZER_KEY,
    SCHEDULER_KEY,
    LOSS_KEY,
    SEED_KEY,
    SCHEDULER_MAX_NUMBER_ITERATIONS_KEY,
    SCHEDULER_MIN_LR_KEY,
    SAVE_MODEL_KEY,
    ENVIRONMENT_KEY,
    MODEL_KEY,
    TASK_KEY,
    MODALITIES_KEY,
    MODALITIES_LABEL_KEY,
    METRICS_CONFIG_KEY,
    METRICS_KEY,
    ESA_WORLD_COVER_INDEX_KEY,
    SENTINEL_2_INDEX_KEY,
    BANDS_KEY,
    DATALOADER_TRAIN_FILE_KEY,
    DATALOADER_VALIDATION_FILE_KEY,
    PIPELINES_CONFIG_KEY,
    SENTINEL_1_INDEX_KEY,
    CLIMATE_ZONE_INDEX_KEY,
    ERA_5_INDEX_KEY,
    SEASON_S2_INDEX_KEY,
    GLO_30_DEM_INDEX_KEY
)

from master_thesis_benge.supervised_baseline.config.config_runs.config_files_and_directories import (
    RemoteFilesAndDirectoryReferences as FileAndDirectoryReferences,
)

from master_thesis_benge.supervised_baseline.training.segmentation.segmentation_utils import (
    SegmentationUtils
)

from remote_sensing_core.constants import Bands

from remote_sensing_core.transforms.ffcv.min_max_scaler import MinMaxScaler
from remote_sensing_core.transforms.ffcv.clipping import Clipping
from remote_sensing_core.transforms.ffcv.channel_selector import ChannelSelector
from remote_sensing_core.transforms.ffcv.add_1d_channel import Add1dChannel
from remote_sensing_core.transforms.ffcv.convert import Convert
from remote_sensing_core.transforms.ffcv.esa_world_cover_transform import EsaWorldCoverTransform
from remote_sensing_core.transforms.ffcv.blow_up import BlowUp
from remote_sensing_core.transforms.ffcv.min_max_scaler import MinMaxScaler
from remote_sensing_core.transforms.ffcv.era5_temperature_s2_transform import Era5TemperatureS2Transform

from ffcv.transforms import ToTensor, ToDevice
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, IntDecoder

training_config = {
    "task": {
        TASK_KEY: Task.SEGMENTATION_ELEVATION.value,
    },
    "model": {
        MODEL_KEY: UNet,
        WEIGHTS_KEY: False,
        NUMBER_OF_CLASSES_KEY: 1,
    },
    "training": {
        MODALITIES_KEY: {
            MODALITIES_LABEL_KEY: GLO_30_DEM_INDEX_KEY,
        },
        #DATALOADER_TRAIN_FILE_KEY: '/ds2/remote_sensing/ben-ge/ffcv/ben-ge-20-train.beton',
        DATALOADER_VALIDATION_FILE_KEY: '/ds2/remote_sensing/ben-ge/ffcv/ben-ge-100-validation.beton',
        EPOCHS_KEY: 20,
        LEARNING_RATE_KEY: 0.01,
        BATCH_SIZE_KEY: 32,
        OPTIMIZER_KEY: torch.optim.Adam,
        #SCHEDULER_KEY: torch.optim.lr_scheduler.CosineAnnealingLR,
        LOSS_KEY: torch.nn.MSELoss(),#weight=[0.1, 10, 0.1, 0.1, 10, 10, 0, 0.1, 10, 0, 10]
        #SEED_KEY: 42,
        SCHEDULER_MAX_NUMBER_ITERATIONS_KEY: 20,
        SCHEDULER_MIN_LR_KEY: 0,
    },
    "metrics": {METRICS_KEY: SegmentationUtils},
    "other": {
        SAVE_MODEL_KEY: False,
        ENVIRONMENT_KEY: "remote",
    },
    "pipelines": {
        'climate_zone': [FloatDecoder(), MinMaxScaler(maximum_value=29, minimum_value=0, interval_max=1, interval_min=0), BlowUp([1,120,120]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        #'elevation_differ': [FloatDecoder(), ToTensor(), ToDevice(device)],
        'era_5': [NDArrayDecoder(), Era5TemperatureS2Transform(), BlowUp([1,120,120]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        #'esa_worldcover': [NDArrayDecoder(), EsaWorldCoverTransform(10,1), Add1dChannel(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'glo_30_dem': [NDArrayDecoder(), ChannelSelector([0]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        #'multiclass_numer': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
        'multiclass_one_h': [ToTensor(), ToDevice(device = torch.device('cuda'))],
        'season_s1': [FloatDecoder(), BlowUp([1,120,120]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'season_s2': [FloatDecoder(), BlowUp([1,120,120]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'sentinel_1': [NDArrayDecoder(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'sentinel_2': [NDArrayDecoder(), Clipping([0, 10_000]), ChannelSelector([7, 3, 2, 1]), ToTensor(), ToDevice(device=torch.device('cuda'))],
    }
    #Modality
    #'esa_worldcover': [NDArrayDecoder(), EsaWorldCoverTransform(), Add1dChannel(), ToTensor(), ToDevice(device = torch.device('cuda'))],
    #Label
    #'esa_worldcover': [NDArrayDecoder(), EsaWorldCoverTransform(), Convert('int64'), ToTensor(), ToDevice(device = torch.device('cuda'))],
}

def get_data_set_files(size: str):
    train_file = f'/ds2/remote_sensing/ben-ge/ffcv/ben-ge-{str(size)}-train.beton'
    validation_file = f'/ds2/remote_sensing/ben-ge/ffcv/ben-ge-{str(size)}-validation.beton'
    return train_file, validation_file