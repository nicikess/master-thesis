from enum import Enum
import torch

# Import transforms
from classification_baseline.transforms import Transforms

# Import models
from classification_baseline.model.resnet import ResNet
from classification_baseline.model.dual_resnet import DualResNet

# Set bands
class Bands(Enum):
    RGB = "RGB"
    INFRARED = "infrared"
    ALL = "all"

# Set training parameters
class TrainingParameters(Enum):
    MODEL = ResNet
    MULTI_MODAL = False
    EPOCHS = 20
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    OPTIMIZER = torch.optim.Adam
    SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR
    LOSS = torch.nn.BCEWithLogitsLoss()
    BANDS = Bands.RGB
    NUMBER_OF_CLASSES = 11
    NUMBER_OF_INPUT_CHANNELS_S1 = 2
    NUMBER_OF_INPUT_CHANNELS_S2 = 3
    SEED = 42
    TRANSFORMS = Transforms().transform
    NORMALIZATION_VALUE = 10_000
    LABEL_THRESHOLD = 0.3
    SCHEDULER_MAX_NUMBER_INTERATIONS = 150_000
    SCHEDULER_MIN_LR = 0

# Set remote file paths and directories
class RemoteFilesAndDirectoryReferences(Enum):
    ESA_WORLD_COVER_CSV_TRAIN = "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-train.csv"
    ESA_WORLD_COVER_CSV_VALIDATION = "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-validation.csv"
    ESA_WORLD_COVER_CSV_TEST = "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-test.csv"
    SENTINEL_1_2_METADATA_CSV = "/ds2/remote_sensing/ben-ge/ben-ge-s/ben-ge-s_sentinel12_meta.csv"
    SENTINEL_1_DIRECTORY = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-1/s1_npy/"
    SENTINEL_2_DIRECTORY = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy/"
    ESA_WORLD_COVER_DIRECTORY = "/ds2/remote_sensing/ben-ge/ben-ge-s/esaworldcover/"
    ERA5_CSV = "/ds2/remote_sensing/ben-ge/ben-ge-s/ben-ge-s_era-5.csv"

# Set local file paths and directories
class LocalFilesAndDirectoryReferences(Enum):
    ESA_WORLD_COVER_CSV_TRAIN = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/data-index/ben-ge-s-train.csv"
    ESA_WORLD_COVER_CSV_VALIDATION = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/data-index/ben-ge-s-validation.csv"
    ESA_WORLD_COVER_CSV_TEST = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/data-index/ben-ge-s-test.csv"
    SENTINEL_1_2_METADATA_CSV = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/ben-ge-s_sentinel12_meta.csv"
    SENTINEL_1_DIRECTORY = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/sentinel-1/s1_npy/"
    SENTINEL_2_DIRECTORY = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/sentinel-2/s2_npy/"
    ESA_WORLD_COVER_DIRECTORY = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/esaworldcover/"
    ERA5_CSV = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/ben-ge-s_era-5.csv"

# Set other values

S1_IMG_KEY = "s1_img"
S2_IMG_KEY = "s2_img"
LABEL_KEY = "label"
NUMPY_DTYPE = "float32"