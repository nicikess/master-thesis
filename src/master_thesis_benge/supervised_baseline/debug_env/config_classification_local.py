import torch

# Import transforms

# Import models
from master_thesis_benge.supervised_baseline.model.resnet import (
    ResNet,
)

from remote_sensing_core.ben_ge.ben_ge_dataset import BenGe

# Import constants
from master_thesis_benge.supervised_baseline.debug_env.constants_local import (
    Task,
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
    SAVE_MODEL_KEY,
    ENVIRONMENT_KEY,
    MODEL_KEY,
    BANDS_KEY,
    MODALITIES_LABEL_KEY,
    MULTICLASS_ONE_HOT_LABEL_KEY,
    MODALITIES_KEY,
    METRICS_KEY,
    S2_MODALITY_KEY
)

from remote_sensing_core.ben_ge.modalities.sentinel_1 import Sentinel1Modality
from remote_sensing_core.ben_ge.modalities.sentinel_2 import (
    Sentinel2Modality,
    Sentinel2Transform,
)
from remote_sensing_core.ben_ge.modalities.esa_worldcover import EsaWorldCoverModality

from master_thesis_benge.supervised_baseline.config.config_runs.config_files_and_directories import (
    LocalFilesAndDirectoryReferences as FileAndDirectoryReferences,
)
from master_thesis_benge.supervised_baseline.training.classification.classification_utils import (
    ClassificationUtils,
)

training_config = {
    "task": {
        TASK_KEY: Task.Classification,
    },
    "model": {
        MODEL_KEY: ResNet,
        WEIGHTS_KEY: False,
        NUMBER_OF_CLASSES_KEY: 11,
    },
    "training": {
        MODALITIES_KEY: {
            MODALITIES_LABEL_KEY: MULTICLASS_ONE_HOT_LABEL_KEY,
            MODALITIES_KEY: [S2_MODALITY_KEY],
        },
        EPOCHS_KEY: 1,
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
        #NORMALIZATION_VALUE_KEY: 10000,
        #LABEL_THRESHOLD_KEY: 0.05,
        BANDS_KEY: "RGB",
        #TRANSFORMS_KEY: Transforms().transform,
    },
    "metrics": {METRICS_KEY: ClassificationUtils},
    "other": {
        SAVE_MODEL_KEY: False,
        ENVIRONMENT_KEY: "local",
    },
}

# Modalities that are loaded
sentinel_1_modality = Sentinel1Modality(
    data_root_path=FileAndDirectoryReferences.SENTINEL_1_DIRECTORY,
    numpy_dtype="float32",
)
sentinel_2_modality = Sentinel2Modality(
    data_root_path=FileAndDirectoryReferences.SENTINEL_2_DIRECTORY,
    s2_bands=training_config["data"][BANDS_KEY],
    # transform=training_config["data"][TRANSFORMS_KEY],
    numpy_dtype="float32",
    transform=Sentinel2Transform(clip_values=(0, 10_000), normalization_value=10_000)
)
esa_world_cover_modality_train = EsaWorldCoverModality(
    data_root_path=FileAndDirectoryReferences.ESA_WORLD_COVER_DIRECTORY,
    esa_world_cover_index_path=FileAndDirectoryReferences.ESA_WORLD_COVER_CSV_TRAIN,
    # transform=training_config["data"][TRANSFORMS_KEY],
)
esa_world_cover_modality_validation = EsaWorldCoverModality(
    data_root_path=FileAndDirectoryReferences.ESA_WORLD_COVER_DIRECTORY,
    esa_world_cover_index_path=FileAndDirectoryReferences.ESA_WORLD_COVER_CSV_VALIDATION,
    # transform=training_config["data"][TRANSFORMS_KEY],
)
modalities_config_train = {
    # "sentinel_1_modality": sentinel_1_modality,
    "sentinel_2_modality": sentinel_2_modality,
    "esa_world_cover_modality": esa_world_cover_modality_train,
}
modalities_config_validation = {
    # "sentinel_1_modality": sentinel_1_modality,
    "sentinel_2_modality": sentinel_2_modality,
    "esa_world_cover_modality": esa_world_cover_modality_validation,
}

dataset_train = BenGe(
    data_index_path=FileAndDirectoryReferences.ESA_WORLD_COVER_CSV_TRAIN,
    sentinel_1_2_metadata_path=FileAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV,
    **modalities_config_train,
)

dataset_validation = BenGe(
    data_index_path=FileAndDirectoryReferences.ESA_WORLD_COVER_CSV_VALIDATION,
    sentinel_1_2_metadata_path=FileAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV,
    **modalities_config_validation,
)