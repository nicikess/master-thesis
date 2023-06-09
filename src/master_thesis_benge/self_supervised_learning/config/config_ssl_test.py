import torch

# Import transforms
from master_thesis_benge.supervised_baseline.training.transforms import (
    Transforms,
)

# Import models
from master_thesis_benge.supervised_baseline.model.resnet import (
    ResNet,
)

from master_thesis_benge.supervised_baseline.model.dual_resnet import (
    DualResNet,
)

from master_thesis_benge.supervised_baseline.model.triple_resnet import (
    TripleResNet,
)

from remote_sensing_core.ben_ge.ben_ge_dataset import BenGe

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
    MODALITIES_LABEL_KEY,
    MODALITIES_KEY,
    MULTICLASS_ONE_HOT_LABEL_INDEX_KEY,
    ESA_WORLD_COVER_INDEX_KEY,
    MODALITIES_KEY,
    METRICS_KEY,
    SENTINEL_2_INDEX_KEY,
    DATALOADER_TRAIN_FILE_KEY,
    DATALOADER_VALIDATION_FILE_KEY,
    TASK_KEY,
)

from master_thesis_benge.supervised_baseline.config.config_runs.config_files_and_directories import (
    RemoteFilesAndDirectoryReferences as FileAndDirectoryReferences,
)
from master_thesis_benge.supervised_baseline.training.classification.classification_utils import (
    ClassificationUtils,
)

from remote_sensing_core.constants import Bands

from remote_sensing_core.transforms.ffcv.min_max_scaler import MinMaxScaler
from remote_sensing_core.transforms.ffcv.clipping import Clipping
from remote_sensing_core.transforms.ffcv.channel_selector import ChannelSelector
from remote_sensing_core.transforms.ffcv.convert import Convert
from remote_sensing_core.transforms.ffcv.esa_world_cover_transform import EsaWorldCoverTransform
from remote_sensing_core.transforms.ffcv.blow_up import BlowUp
from remote_sensing_core.transforms.ffcv.min_max_scaler import MinMaxScaler
from remote_sensing_core.transforms.ffcv.era5_temperature_s2_transform import Era5TemperatureS2Transform

from ffcv.transforms import ToTensor, ToDevice
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, IntDecoder

training_config = {
    "pipelines": {
        'climate_zone': None,
        'elevation_differ': None,
        'era_5': None,
        'esa_worldcover': None,
        'glo_30_dem': None,
        'multiclass_numer': None,
        'multiclass_one_h': None,
        'season_s1': None,
        'season_s2': None,
        'sentinel_1': [NDArrayDecoder(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'sentinel_2': [NDArrayDecoder(), Clipping([0, 10_000]), ChannelSelector([7, 3, 2, 1]), ToTensor(), ToDevice(device=torch.device('cuda'))],
        'field_names': None,
    },
}


def get_data_set_files(size: str):
    train_file = f'/raid/remote_sensing/ben-ge/ffcv/ben-ge-{str(size)}-train.beton'
    validation_file = f'/raid/remote_sensing/ben-ge/ffcv/ben-ge-{str(size)}-validation.beton'
    return train_file, validation_file
