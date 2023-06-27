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

from master_thesis_benge.supervised_baseline.model.triple_unet import (
    TripleUNet,
)

# Import constants
from master_thesis_benge.supervised_baseline.config.constants import (
    Task,
    NUMBER_OF_CLASSES_KEY,
    EPOCHS_KEY,
    LEARNING_RATE_KEY,
    BATCH_SIZE_KEY,
    OPTIMIZER_KEY,
    SCHEDULER_KEY,
    LOSS_KEY,
    SCHEDULER_MAX_NUMBER_ITERATIONS_KEY,
    SCHEDULER_MIN_LR_KEY,
    SAVE_MODEL_KEY,
    MODEL_KEY,
    TASK_KEY,
    MODALITIES_KEY,
    MODALITIES_LABEL_KEY,
    METRICS_KEY,
    DATALOADER_TRAIN_FILE_KEY,
    DATALOADER_VALIDATION_FILE_KEY,
    GLO_30_DEM_INDEX_KEY,
)

from master_thesis_benge.supervised_baseline.config.config_runs.config_files_and_directories import (
    RemoteFilesAndDirectoryReferences as FileAndDirectoryReferences,
)

from master_thesis_benge.supervised_baseline.training.regression.regression_utils import (
    RegressionUtils
)

from remote_sensing_core.constants import Bands

from remote_sensing_core.transforms.ffcv.min_max_scaler import MinMaxScaler
from remote_sensing_core.transforms.ffcv.clipping import Clipping
from remote_sensing_core.transforms.ffcv.channel_selector import ChannelSelector
from remote_sensing_core.transforms.ffcv.convert import Convert
from remote_sensing_core.transforms.ffcv.blow_up import BlowUp
from remote_sensing_core.transforms.ffcv.min_max_scaler import MinMaxScaler
from remote_sensing_core.transforms.ffcv.era5_temperature_s2_transform import Era5TemperatureS2Transform
from remote_sensing_core.transforms.ffcv.esa_world_cover_transform import EsaWorldCoverTransform
from remote_sensing_core.transforms.ffcv.expand_dimension import ExpandDimension


from ffcv.transforms import ToTensor, ToDevice
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, IntDecoder

training_config = {
    "task": {
        TASK_KEY: Task.SEGMENTATION_ELEVATION.value,
    },
    "model": {
        MODEL_KEY: TripleUNet,
        NUMBER_OF_CLASSES_KEY: 1, #Is not used in the code, but if is not set, the code will crash (because train requires it)
    },
    "training": {
        MODALITIES_KEY: {
            MODALITIES_LABEL_KEY: GLO_30_DEM_INDEX_KEY,
        },
        DATALOADER_TRAIN_FILE_KEY: '/raid/remote_sensing/ben-ge/ffcv/ben-ge-20-train.beton',
        DATALOADER_VALIDATION_FILE_KEY: '/raid/remote_sensing/ben-ge/ffcv/ben-ge-20-validation.beton',
        EPOCHS_KEY: 20,
        LEARNING_RATE_KEY: 0.0001,
        BATCH_SIZE_KEY: 16,
        OPTIMIZER_KEY: torch.optim.Adam,
        SCHEDULER_KEY: torch.optim.lr_scheduler.CosineAnnealingLR,
        LOSS_KEY: torch.nn.MSELoss(),#weight=[0.1, 10, 0.1, 0.1, 10, 10, 0, 0.1, 10, 0, 10]
        #SEED_KEY: 42,
        SCHEDULER_MAX_NUMBER_ITERATIONS_KEY: 20,
        SCHEDULER_MIN_LR_KEY: 0
    },
    "metrics": {METRICS_KEY: RegressionUtils},
    "other": {
        SAVE_MODEL_KEY: False,
    },
    "pipelines": {
        'climate_zone': [FloatDecoder(), MinMaxScaler(minimum_value=0, maximum_value=29, interval_min=0, interval_max=1), BlowUp([1,120,120]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        #'elevation_differ': [FloatDecoder(), ToTensor(), ToDevice(device)],
        'era_5': [NDArrayDecoder(), Era5TemperatureS2Transform(batch_size=16), BlowUp([1,120,120]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'esa_worldcover': [NDArrayDecoder(), EsaWorldCoverTransform(10,1), ExpandDimension(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'glo_30_dem': [NDArrayDecoder(), ChannelSelector([0]), Clipping([0, 500]), MinMaxScaler(maximum_value=500, minimum_value=0, interval_max=1, interval_min=0), ToTensor(), ToDevice(device = torch.device('cuda'))],
        #'multiclass_numer': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
        'multiclass_one_h': [ToTensor(), ToDevice(device = torch.device('cuda'))],
        'season_s1': [FloatDecoder(), BlowUp([1,120,120]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'season_s2': [FloatDecoder(), BlowUp([1,120,120]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'sentinel_1': [NDArrayDecoder(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'sentinel_2': [NDArrayDecoder(), Clipping([0, 10_000]), ChannelSelector([7, 3, 2, 1]), ToTensor(), ToDevice(device=torch.device('cuda'))],
    }
}

def get_data_set_files(size: str):
    train_file = f'/raid/remote_sensing/ben-ge/ffcv/ben-ge-{str(size)}-train.beton'
    validation_file = f'/raid/remote_sensing/ben-ge/ffcv/ben-ge-{str(size)}-validation.beton'
    #return train_file, validation_file