import torch

from remote_sensing_core.ben_ge.ben_ge_dataset import BenGe

# Import constants
from master_thesis_benge.self_supervised_learning.config.constants import (
    EPOCHS_KEY,
    LEARNING_RATE_KEY,
    SEED_KEY,
    SAVE_MODEL_KEY,
    GRADIENT_ACCUMULATION_STEPS_KEY,
    WEIGHT_DECAY_KEY,
    EMEDDING_SIZE_KEY,
    CHECKPOINT_PATH_KEY,
    DATASET_SIZE_KEY,
    FEATURE_DIMENSION_KEY,
    RESUME_FROM_CHECKPOINT_KEY
)

from remote_sensing_core.constants import Bands

from remote_sensing_core.transforms.ffcv.min_max_scaler import MinMaxScaler
from remote_sensing_core.transforms.ffcv.clipping import Clipping
from remote_sensing_core.transforms.ffcv.channel_selector import ChannelSelector
from remote_sensing_core.transforms.ffcv.convert import Convert
from remote_sensing_core.transforms.ffcv.esa_world_cover_transform import EsaWorldCoverTransform
from remote_sensing_core.transforms.ffcv.blow_up import BlowUp
from remote_sensing_core.transforms.ffcv.min_max_scaler import MinMaxScaler
from remote_sensing_core.transforms.ffcv.expand_dimension import ExpandDimension

from ffcv.transforms import ToTensor, ToDevice
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, IntDecoder

training_config = {
    "training": {
        EPOCHS_KEY: 50,
        SEED_KEY: 42,
        LEARNING_RATE_KEY: 3e-4,
        #IMAGE_SIZE_KEY: 120, -> use for augmentation
        SAVE_MODEL_KEY: "saved_models/",
        GRADIENT_ACCUMULATION_STEPS_KEY: 5,
        WEIGHT_DECAY_KEY: 1e-6,
        EMEDDING_SIZE_KEY: 128,
        CHECKPOINT_PATH_KEY: "-",
        #DATASET_SIZE_KEY: "20",
        FEATURE_DIMENSION_KEY: 512,
        RESUME_FROM_CHECKPOINT_KEY: False,
    },
    "pipelines": {
        #'climate_zone': [FloatDecoder(), MinMaxScaler(maximum_value=29, minimum_value=0, interval_max=1, interval_min=0), BlowUp([1,120,120]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        #'elevation_differ': [FloatDecoder(), ToTensor(), ToDevice(device)],
        #'era_5': [NDArrayDecoder(), Era5TemperatureS2Transform(batch_size=32), BlowUp([1,120,120]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'esa_worldcover': [NDArrayDecoder(), EsaWorldCoverTransform(10,1), ExpandDimension(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'glo_30_dem': [NDArrayDecoder(), ChannelSelector([0]), ToTensor(), ToDevice(device = torch.device('cuda'))],
        #'multiclass_numer': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
        #'multiclass_one_h': [ToTensor(), ToDevice(device = torch.device('cuda'))],
        #'season_s1': [FloatDecoder(), BlowUp([1,120,120]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        #'season_s2': [FloatDecoder(), BlowUp([1,120,120]), Convert('float32'), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'sentinel_1': [NDArrayDecoder(), ToTensor(), ToDevice(device = torch.device('cuda'))],
        'sentinel_2': [NDArrayDecoder(), Clipping([0, 10_000]), ChannelSelector([7, 3, 2, 1]), ToTensor(), ToDevice(device=torch.device('cuda'))],
    },
}

def get_data_set_files(size: str):
    train_file = f'/raid/remote_sensing/ben-ge/ffcv/ben-ge-{str(size)}-train.beton'
    validation_file = f'/raid/remote_sensing/ben-ge/ffcv/ben-ge-{str(size)}-validation.beton'
    return train_file, validation_file
