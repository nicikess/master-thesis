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


# Modalities
S1_MODALITY_KEY = "sentinel_1"
S2_MODALITY_KEY = "sentinel_2"
WORLD_COVER_MODALITY_KEY = "esa_worldcover"
ALTITUDE_MODALITY_KEY = "glo_30_dem"
ERA_5_MODALITY_KEY = "era_5"
STACKED_IMAGE_KEY = "stacked_img"

# Labels
MULTICLASS_NUMERIC_LABEL_KEY = "multiclass_numeric_label"
MULTICLASS_ONE_HOT_LABEL_KEY = "multiclass_one_hot_label"

# Task
TASK_KEY = "task"

# Data
DATASET_TRAIN_KEY = "dataset_train"
DATASET_VALIDATION_KEY = "dataset_validation"

# Training
TRAIN_CLASS_KEY = "train_class"
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
TRANSFORMS_KEY = "transforms"
TRANSFORMS_LABEL_KEY = "transforms_label"
NORMALIZATION_VALUE_KEY = "normalization_value"
LABEL_THRESHOLD_KEY = "label_threshold"
SCHEDULER_MAX_NUMBER_ITERATIONS_KEY = "scheduler_max_number_iterations"
SCHEDULER_MIN_LR_KEY = "scheduler_min_lr"
SAVE_MODEL_KEY = "save_model"

# Environment
ENVIRONMENT_KEY = "environment"

# Metrics
METRICS_KEY = "metrics"

# Config
DATA_CONFIG_KEY = "data"
LABEL_CONFIG_KEY = "label"
TASK_CONFIG_KEY = "task"
TRAINING_CONFIG_KEY = "training"
MODEL_CONFIG_KEY = "model"
MODALITIES_CONFIG_KEY = "modalities"
OTHER_CONFIG_KEY = "other"
METRICS_CONFIG_KEY = "metrics"