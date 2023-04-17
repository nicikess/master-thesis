from enum import Enum


# Set bands
class Bands(Enum):
    RGB = "RGB"
    INFRARED = "infrared"
    ALL = "all"


class Task(Enum):
    Classification = ("classification",)
    Regression = ("regression",)
    Segmentation = "segmentation"


# Modalities
S1_MODALITY_KEY = "sentinel_1_modality"
S2_MODALITY_KEY = "sentinel_2_modality"
WORLD_COVER_MODALITY_KEY = "esa_worldcover_modality"
ALTITUDE_MODALITY_KEY = "glo_30_dem"
ERA_5_MODALITY_KEY = "era_5"
STACKED_IMAGE_KEY = "stacked_img"

# Labels
MULTICLASS_LABEL_KEY = "multiclass_label"

# Task
TASK_KEY = "task"

# Data
DATASET_TRAIN_KEY = "dataset_train"
DATASET_VALIDATION_KEY = "dataset_validation"

# Training
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
NORMALIZATION_VALUE_KEY = "normalization_value"
LABEL_THRESHOLD_KEY = "label_threshold"
SCHEDULER_MAX_NUMBER_ITERATIONS_KEY = "scheduler_max_number_iterations"
SCHEDULER_MIN_LR_KEY = "scheduler_min_lr"
SAVE_MODEL_KEY = "save_model"

# Environment
ENVIRONMENT_KEY = "environment"

# Config
DATA_CONFIG_KEY = "data"
TRAINING_CONFIG_KEY = "training"
MODEL_CONFIG_KEY = "model"
MODALITIES_CONFIG_KEY = "modalities"
OTHER_CONFIG_KEY = "other"
