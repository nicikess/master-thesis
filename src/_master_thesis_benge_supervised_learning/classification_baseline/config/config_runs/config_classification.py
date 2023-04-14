import torch

# Import transforms
from _master_thesis_benge_supervised_learning.classification_baseline.training.transforms import (
    Transforms,
)

# Import models
from _master_thesis_benge_supervised_learning.classification_baseline.model.dual_resnet import (
    DualResNet,
)

# Import constants
from _master_thesis_benge_supervised_learning.classification_baseline.config.constants import (
    TASK_KEY,
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
    NORMALIZATION_VALUE_KEY,
    LABEL_THRESHOLD_KEY,
    SAVE_MODEL_KEY,
    ENVIRONMENT_KEY,
    Task,
    MODEL_KEY,
    BANDS_KEY,
    TRANSFORMS_KEY,
)

from remote_sensing_core.ben_ge.modalities.sentinel_1 import Sentinel1Modality
from remote_sensing_core.ben_ge.modalities.sentinel_2 import Sentinel2Modality
from remote_sensing_core.ben_ge.modalities.esa_worldcover import EsaWorldCoverModality

from _master_thesis_benge_supervised_learning.classification_baseline.config.config_runs.config_files_and_directories import (
    LocalFilesAndDirectoryReferences,
)

training_config = {
    "task": {
        TASK_KEY: Task.Classification,
    },
    "model": {
        MODEL_KEY: DualResNet,
        WEIGHTS_KEY: False,
        NUMBER_OF_CLASSES_KEY: 11,
    },
    "training": {
        EPOCHS_KEY: 20,
        LEARNING_RATE_KEY: 0.001,
        BATCH_SIZE_KEY: 32,
        OPTIMIZER_KEY: torch.optim.Adam,
        SCHEDULER_KEY: torch.optim.lr_scheduler.CosineAnnealingLR,
        LOSS_KEY: torch.nn.BCEWithLogitsLoss(),
        SEED_KEY: 42,
        SCHEDULER_MAX_NUMBER_ITERATIONS_KEY: 20,
        SCHEDULER_MIN_LR_KEY: 0,
    },
    "data": {
        NORMALIZATION_VALUE_KEY: 10000,
        LABEL_THRESHOLD_KEY: 0.05,
        BANDS_KEY: "RGB",
        TRANSFORMS_KEY: Transforms().transform,
    },
    "other": {
        SAVE_MODEL_KEY: False,
        ENVIRONMENT_KEY: "local",
    },
}

# Modalities
sentinel_1_modality = Sentinel1Modality(
    data_root_path=LocalFilesAndDirectoryReferences.SENTINEL_1_DIRECTORY,
)
sentinel_2_modality = Sentinel2Modality(
    data_root_path=LocalFilesAndDirectoryReferences.SENTINEL_2_DIRECTORY,
    s2_bands=training_config["data"][BANDS_KEY],
    transform=training_config["data"][TRANSFORMS_KEY],
)
esa_world_cover_modality = EsaWorldCoverModality(
    data_root_path=LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_DIRECTORY,
    esa_world_cover_index_path=LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_TRAIN,
)
modalities_config = {
    "sentinel_1_modality": sentinel_1_modality,
    "sentinel_2_modality": sentinel_2_modality,
    "esa_world_cover_modality": esa_world_cover_modality,
}
