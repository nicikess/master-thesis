import torch

# Import transforms
from master_thesis_benge.supervised_baseline.training.transforms import (
    Transforms,
)

# Import models
from master_thesis_benge.supervised_baseline.model.resnet import (
    ResNet,
)

# Import constants
from master_thesis_benge.supervised_baseline.config.constants import (
    Task,
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
    MODALITIES_KEY,
    MODALITIES_LABEL_KEY,
    METRICS_KEY,
    MULTICLASS_ONE_HOT_LABEL_INDEX_KEY,
    TASK_KEY,
    SENTINEL_2_INDEX_KEY
)

from master_thesis_benge.supervised_baseline.config.config_runs.config_files_and_directories import (
    RemoteFilesAndDirectoryReferences as FileAndDirectoryReferences,
)
from master_thesis_benge.supervised_baseline.training.regression.regression_utils import (
    RegressionUtils
)

training_config = {
    "task": {
        TASK_KEY: Task.Regression,
    },
    "model": {
        MODEL_KEY: ResNet,
        WEIGHTS_KEY: False,
        NUMBER_OF_CLASSES_KEY: 11,
    },
    "training": {
        MODALITIES_KEY: {
            MODALITIES_LABEL_KEY: MULTICLASS_ONE_HOT_LABEL_INDEX_KEY,
            MODALITIES_KEY: [SENTINEL_2_INDEX_KEY],
        },
        EPOCHS_KEY: 20,
        LEARNING_RATE_KEY: 0.001,
        BATCH_SIZE_KEY: 32,
        OPTIMIZER_KEY: torch.optim.Adam,
        SCHEDULER_KEY: torch.optim.lr_scheduler.CosineAnnealingLR,
        LOSS_KEY: torch.nn.MSELoss(),
        SEED_KEY: 42,
        SCHEDULER_MAX_NUMBER_ITERATIONS_KEY: 20,
        SCHEDULER_MIN_LR_KEY: 0,
    },
    "metrics": {METRICS_KEY: RegressionUtils},
    "other": {
        SAVE_MODEL_KEY: False,
        ENVIRONMENT_KEY: "remote",
    },
}