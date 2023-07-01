from enum import Enum

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

label_from_index = {
    0: "climatezone",
    1: "elevation-difference-label",
    2: "temperature(era5)",
    3: "worldcover(esa)",
    4: "elevation(glo-30-dem)",
    5: "multiclass-numeric-label",
    6: "multiclass-one-hot-label",
    7: "season(s1)",
    8: "season(s2)",
    9: "sentinel1",
    10: "sentinel2"
}

class Task(Enum):
    SSL_CLASSIFICATION_LANDUSE_MULTILABEL = "ssl-classification-landuse-multilabel"
    SSL_SEGMENTATION_LANDUSE = "ssl-segmentation-landuse"

# Training
EPOCHS_KEY = "epochs"
SEED_KEY = "seed"
LEARNING_RATE_KEY = "learning_rate"
SAVE_MODEL_KEY = "save_model"
DATALOADER_TRAIN_FILE_KEY = "dataloader_train_file"
DATALOADER_VALIDATION_FILE_KEY = "dataloader_validation_file"
DATASET_SIZE_KEY = "dataset_size"
IMAGE_SIZE_KEY = "image_size"
GRADIENT_ACCUMULATION_STEPS_KEY = "gradient_accumulation_steps"
WEIGHT_DECAY_KEY = "weight_decay"
EMEDDING_SIZE_KEY = "embedding_size"
CHECKPOINT_PATH_KEY = "checkpoint_path"
NUMBER_OF_CLASSES_KEY = "number_of_classes"
BATCH_SIZE_KEY = "batch_size"
OPTIMIZER_KEY = "optimizer"
SCHEDULER_KEY = "scheduler"
LOSS_KEY = "loss"
SCHEDULER_MAX_NUMBER_ITERATIONS_KEY = "scheduler_max_number_iterations"
SCHEDULER_MIN_LR_KEY = "scheduler_min_lr"
MODEL_KEY = "model"
TASK_KEY = "task"
MODALITIES_KEY = "modalities"
MODALITIES_LABEL_KEY = "modalities_label"
FEATURE_DIMENSION_KEY = "feature_dimension"
RESUME_FROM_CHECKPOINT_KEY = "resume_from_checkpoint"
PROJECTION_HEAD_KEY = "projection_head"

# Metrics
METRICS_KEY = "metrics"

# Config
TRAINING_CONFIG_KEY = "training"
PIPELINES_CONFIG_KEY = "pipelines"
PARAMETERS_CONFIG_KEY = "parameters"
TRAINING_RESNET_CONFIG_KEY = "resnet"
TRAINING_UNET_CONFIG_KEY = "unet"
EVALUATION_CLASSIFICATION_LANDUSE_MULTILABEL_CONFIG_KEY = "evaluation_config_classification_landuse_multilabel"
EVALUATION_SEGMENTATION_LANDUSE_CONFIG_KEY = "evaluation_config_segmentation_landuse"

# Config
TRAINING_CONFIG_KEY = "training"
TASK_CONFIG_KEY = "task"
MODEL_CONFIG_KEY = "model"
OTHER_CONFIG_KEY = "other"
METRICS_CONFIG_KEY = "metrics"

def get_label_from_index(index: int):
    return label_from_index.get(index)