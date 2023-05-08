import torch

# Import transforms
from _master_thesis_benge_supervised_learning.supervised_baseline.training.transforms import (
    Transforms,
)

# Import models
from _master_thesis_benge_supervised_learning.supervised_baseline.model.unet import (
    UNet,
)

# Import constants
from _master_thesis_benge_supervised_learning.supervised_baseline.config.constants import (
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
    PIPELINES_CONFIG_KEY
)

from _master_thesis_benge_supervised_learning.supervised_baseline.config.config_runs.config_files_and_directories import (
    RemoteFilesAndDirectoryReferences as FileAndDirectoryReferences,
)

from _master_thesis_benge_supervised_learning.supervised_baseline.training.segmentation.segmentation_utils import (
    SegmentationUtils
)

from remote_sensing_core.constants import Bands

from remote_sensing_core.transforms.ffcv.min_max_scaler import MinMaxScaler
from remote_sensing_core.transforms.ffcv.clipping import Clipping
from remote_sensing_core.transforms.ffcv.channel_selector import ChannelSelector
from remote_sensing_core.transforms.ffcv.add_1d_channel import Add1dChannel
from remote_sensing_core.transforms.ffcv.convert import Convert
from remote_sensing_core.transforms.ffcv.esa_world_cover_transform import EsaWorldCoverTransform

from ffcv.transforms import ToTensor, ToDevice
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, IntDecoder

training_config = {
    "task": {
        TASK_KEY: Task.Segmentation,
    },
    "model": {
        MODEL_KEY: UNet,
        WEIGHTS_KEY: False,
        NUMBER_OF_CLASSES_KEY: 11,
    },
    "training": {
        MODALITIES_KEY: {
            MODALITIES_LABEL_KEY: ESA_WORLD_COVER_INDEX_KEY,
            MODALITIES_KEY: [SENTINEL_2_INDEX_KEY],
        },
        DATALOADER_TRAIN_FILE_KEY: '/ds2/remote_sensing/ben-ge/ffcv/ben-ge-train.beton',
        DATALOADER_VALIDATION_FILE_KEY: '/ds2/remote_sensing/ben-ge/ffcv/ben-ge-validation.beton',
        EPOCHS_KEY: 20,
        LEARNING_RATE_KEY: 0.01,
        BATCH_SIZE_KEY: 32,
        OPTIMIZER_KEY: torch.optim.Adam,
        SCHEDULER_KEY: torch.optim.lr_scheduler.CosineAnnealingLR,
        LOSS_KEY: torch.nn.CrossEntropyLoss(),
        SEED_KEY: 42,
        SCHEDULER_MAX_NUMBER_ITERATIONS_KEY: 20,
        SCHEDULER_MIN_LR_KEY: 0,
    },
    "metrics": {METRICS_KEY: SegmentationUtils},
    "other": {
        SAVE_MODEL_KEY: False,
        ENVIRONMENT_KEY: "remote",
    },
    "pipelines": {
        'climate_zone': [FloatDecoder(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        #'elevation_differ': [FloatDecoder(), ToTensor(), ToDevice(device)],
        'era_5': [NDArrayDecoder(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'esa_worldcover': [NDArrayDecoder(), EsaWorldCoverTransform(10,1), Convert('int64'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'glo_30_dem': [NDArrayDecoder(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        #'multiclass_numer': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
        'multiclass_one_h': [ToTensor(), ToDevice(device = torch.device('cuda'))],
        'season_s1': [FloatDecoder(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'season_s2': [FloatDecoder(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'sentinel_1': [NDArrayDecoder(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'sentinel_2': [NDArrayDecoder(), Clipping([0, 10_000]), ChannelSelector([7, 3, 2, 1]), ToTensor(), ToDevice(device = torch.device('cuda'))], 
    }
    #RGB
    #[3, 2, 1]

    #Modality
    #'esa_worldcover': [NDArrayDecoder(), EsaWorldCoverTransform(), Add1dChannel(), ToTensor(), ToDevice(device = torch.device('cuda'))],
    
    #Label
    #'esa_worldcover': [NDArrayDecoder(), EsaWorldCoverTransform(), Convert('int64'), ToTensor(), ToDevice(device = torch.device('cuda'))],


}