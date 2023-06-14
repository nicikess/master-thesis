

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

# Training
EPOCHS_KEY = "epochs"
SEED_KEY = "seed"
LEARNING_RATE_KEY = "learning_rate"
SAVE_MODEL_KEY = "save_model"
DATALOADER_TRAIN_FILE_KEY = "dataloader_train_file"
DATALOADER_VALIDATION_FILE_KEY = "dataloader_validation_file"
DATASET_SIZE_KEY = "dataset_size"
DEVICE_KEY = "device"
IMAGE_SIZE_KEY = "image_size"
LOAD_MODEL_KEY = "load_model"
GRADIENT_ACCUMULATION_STEPS_KEY = "gradient_accumulation_steps"
WEIGHT_DECAY_KEY = "weight_decay"
EMEDDING_SIZE_KEY = "embedding_size"
CHECKPOINT_PATH_KEY = "checkpoint_path"
DATASET_SIZE_KEY = "dataset_size"

# Environment
ENVIRONMENT_KEY = "environment"

# Metrics
METRICS_KEY = "metrics"

# Config
TRAINING_CONFIG_KEY = "training"
PIPELINES_CONFIG_KEY = "pipelines"
PARAMETERS_CONFIG_KEY = "parameters"

# Main
MODE_TRAIN_KEY = "train"
MODE_EVALUATE_KEY = "evaluate"
MODE_SAVE_WEIGHTS_KEY = "save_weights"

def get_label_from_index(index: int):
    return label_from_index.get(index)