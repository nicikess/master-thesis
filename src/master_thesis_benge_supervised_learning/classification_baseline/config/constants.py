from enum import Enum

# Set bands
class Bands(Enum):
    RGB = "RGB"
    INFRARED = "infrared"
    ALL = "all"

# Set remote file paths and directories for ben-ge (small)
class RemoteFilesAndDirectoryReferencesSmall():
    # Files
    ESA_WORLD_COVER_CSV_TRAIN = "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-train.csv"
    ESA_WORLD_COVER_CSV_VALIDATION = "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-validation.csv"
    ESA_WORLD_COVER_CSV_TEST = "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-test.csv"
    SENTINEL_1_2_METADATA_CSV = "/ds2/remote_sensing/ben-ge/ben-ge-s/ben-ge-s_sentinel12_meta.csv"

    # Directories
    SENTINEL_1_DIRECTORY = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-1/s1_npy/"
    SENTINEL_2_DIRECTORY = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy/"
    ESA_WORLD_COVER_DIRECTORY = "/ds2/remote_sensing/ben-ge/ben-ge-s/esaworldcover/npy"
    ERA5_CSV = "/ds2/remote_sensing/ben-ge/ben-ge-s/ben-ge-s_era-5.csv"
    GLO_30_DIRECTORY = '/ds2/remote_sensing/ben-ge/ben-ge-s/glo-30_dem'

# Set remote file paths and directories for ben-ge (small)
class RemoteFilesAndDirectoryReferencesLarge():
    # Files
    ESA_WORLD_COVER_CSV_TRAIN = "/ds2/remote_sensing/ben-ge/ben-ge/data-index/ben-ge-train.csv"
    ESA_WORLD_COVER_CSV_VALIDATION = "/ds2/remote_sensing/ben-ge/ben-ge/data-index/ben-ge-validation.csv"
    ESA_WORLD_COVER_CSV_TEST = "/ds2/remote_sensing/ben-ge/ben-ge/data-index/ben-ge-test.csv"
    SENTINEL_1_2_METADATA_CSV = "/ds2/remote_sensing/ben-ge/ben-ge/ben-ge_meta.csv"

    # Directories
    SENTINEL_1_DIRECTORY = "/ds2/remote_sensing/ben-ge/ben-ge/sentinel-1/"
    SENTINEL_2_DIRECTORY = "/ds2/remote_sensing/ben-ge/ben-ge/sentinel-2/"
    ESA_WORLD_COVER_DIRECTORY = "/ds2/remote_sensing/ben-ge/ben-ge/esaworldcover/npy"
    ERA5_CSV = "/ds2/remote_sensing/ben-ge/ben-ge/era-5/ben-ge_era-5.csv"
    GLO_30_DIRECTORY = '/ds2/remote_sensing/ben-ge/ben-ge/glo-30_dem'

# Set local file paths and directories (large)
class LocalFilesAndDirectoryReferences():
    ESA_WORLD_COVER_CSV_TRAIN = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/data-index/ben-ge-s-train.csv"
    ESA_WORLD_COVER_CSV_VALIDATION = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/data-index/ben-ge-s-validation.csv"
    ESA_WORLD_COVER_CSV_TEST = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/data-index/ben-ge-s-test.csv"

    SENTINEL_1_2_METADATA_CSV = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/ben-ge-s_sentinel12_meta.csv"
    SENTINEL_1_DIRECTORY = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/sentinel-1/s1_npy/"
    SENTINEL_2_DIRECTORY = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/sentinel-2/s2_npy/"
    ESA_WORLD_COVER_DIRECTORY = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/esaworldcover/"
    ERA5_CSV = "/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/ben-ge-s_era-5.csv"
    GLO_30_DIRECTORY = '/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/glo-30_dem'

# Images
S1_IMG_KEY = "s1_img"
S2_IMG_KEY = "s2_img"
WORLD_COVER_IMG_KEY = "world_cover_img"
ALTITUDE_IMG_KEY = "altitude_img"
STACKED_IMAGE_KEY = "stacked_img"
# Labels
MULTICLASS_LABEL_KEY = "multiclass_label"
# File data type
NUMPY_DTYPE = "float32"
# Training
MODEL_KEY = "model"
MODEL_SIZE_KEY = "model_size"
MULTI_MODAL_KEY = "multi_modal"
WEIGHTS_KEY = "weights"
EPOCHS_KEY = "epochs"
LEARNING_RATE_KEY = "learning_rate"
BATCH_SIZE_KEY = "batch_size"
OPTIMIZER_KEY = "optimizer"
SCHEDULER_KEY = "scheduler"
LOSS_KEY = "loss"
BANDS_KEY = "bands"
NUMBER_OF_CLASSES_KEY = "number_of_classes"
NUMBER_OF_INPUT_CHANNELS_S1_KEY = "number_of_input_channels_s1"
NUMBER_OF_INPUT_CHANNELS_S2_KEY = "number_of_input_channels_s2"
SEED_KEY = "seed"
TRANSFORMS_KEY = "transforms"
NORMALIZATION_VALUE_KEY = "normalization_value"
LABEL_THRESHOLD_KEY = "label_threshold"
SCHEDULER_MAX_NUMBER_ITERATIONS_KEY = "scheduler_max_number_iterations"
SCHEDULER_MIN_LR_KEY = "scheduler_min_lr"
SAVE_MODEL_KEY = "save_model"
SHUFFLE_TRAINING_DATA_KEY = "shuffle_training_data"
SHUFFLE_VALIDATION_DATA_KEY = "shuffle_validation_data"
DATA_SET_SIZE_SMALL_KEY = "data_set_size_small"
ENVIRONMENT_KEY = "environment"

# Config keys
DATA_CONFIG_KEY = "data"
TRAINING_CONFIG_KEY = "training"
MODEL_CONFIG_KEY = "model"
OTHER_CONFIG_KEY = "other"