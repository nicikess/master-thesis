import torch

from remote_sensing_core.ben_ge.ben_ge_dataset import BenGe

# Import constants
from master_thesis_benge.self_supervised_learning.config.constants import (
    EPOCHS_KEY,
    LEARNING_RATE_KEY,
    SEED_KEY,
    SAVE_MODEL_KEY,
    DEVICE_KEY,
    IMAGE_SIZE_KEY,
    LOAD_MODEL_KEY,
    GRADIENT_ACCUMULATION_STEPS_KEY,
    WEIGHT_DECAY_KEY,
    EMEDDING_SIZE_KEY,
    CHECKPOINT_PATH_KEY,
    DATASET_SIZE_KEY,
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
    "parameters": {
        EPOCHS_KEY: 50,
        SEED_KEY: 42,
        LEARNING_RATE_KEY: 3e-4,
        DEVICE_KEY: torch.device('cuda'),
        IMAGE_SIZE_KEY: 120,
        SAVE_MODEL_KEY: "./saved_models/",
        LOAD_MODEL_KEY: False,
        GRADIENT_ACCUMULATION_STEPS_KEY: 5,
        WEIGHT_DECAY_KEY: 1e-6,
        EMEDDING_SIZE_KEY: 128,
        CHECKPOINT_PATH_KEY: "./SimCLR_ResNet18.ckpt",
        DATASET_SIZE_KEY: "20",
    },
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
