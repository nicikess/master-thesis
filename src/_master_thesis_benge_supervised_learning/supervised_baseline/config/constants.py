from enum import Enum


# Set bands
class Bands(Enum):
    RGB = "RGB"
    INFRARED = "infrared"
    ALL = "all"


class Task(Enum):
    Classification = "classification"
    Regression = "regression"
    Segmentation = "segmentation"

# Modalities indicis
CLIMATE_ZONE_INDEX_KEY = 0
ELEVATION_DIFFERENCE_LABEL_INDEX_KEY = 1
ERA_5_INDEX_KEY = 2
ESA_WORLD_COVER_INDEX_KEY = 3
GLO_30_DEM_INDEX_KEY = 4
MULTICLASS_NUMERIC_LABEL_INDEX_KEY = 5
MULTICLASS_ONE_HOT_LABEL_INDEX_KEY = 6
SEASON_S1_INDEX_KEY = 7
SEASON_S2_INDEX_KEY = 8
SENTINEL_1_INDEX_KEY = 9
SENTINEL_2_INDEX_KEY = 10

# Training
DATALOADER_TRAIN_FILE_KEY = "dataloader_train_file"
DATALOADER_VALIDATION_FILE_KEY = "dataloader_validation_file"
MODALITIES_KEY = "modalities"
MODALITIES_LABEL_KEY = "modalities_label"
MODEL_KEY = "model"
WEIGHTS_KEY = "weights"
EPOCHS_KEY = "epochs"
LEARNING_RATE_KEY = "learning_rate"
BATCH_SIZE_KEY = "batch_size"
OPTIMIZER_KEY = "optimizer"
SCHEDULER_KEY = "scheduler"
LOSS_KEY = "loss"
BANDS_KEY = "bands"
NUMBER_OF_CLASSES_KEY = "number_of_classes"
SEED_KEY = "seed"
SCHEDULER_MAX_NUMBER_ITERATIONS_KEY = "scheduler_max_number_iterations"
SCHEDULER_MIN_LR_KEY = "scheduler_min_lr"
SAVE_MODEL_KEY = "save_model"
TASK_KEY = "task"
BANDS_KEY = "bands"

# Environment
ENVIRONMENT_KEY = "environment"

# Metrics
METRICS_KEY = "metrics"

# Config
TRAINING_CONFIG_KEY = "training"
LABEL_CONFIG_KEY = "label"
TASK_CONFIG_KEY = "task"
MODEL_CONFIG_KEY = "model"
OTHER_CONFIG_KEY = "other"
METRICS_CONFIG_KEY = "metrics"
PIPELINES_CONFIG_KEY = "pipelines"