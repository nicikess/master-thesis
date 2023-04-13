from enum import Enum

import pandas as pd
import torch
import torchvision.models as models

# Import transforms
from master_thesis_benge_supervised_learning.classification_baseline.training.transforms import Transforms

# Import models
from master_thesis_benge_supervised_learning.classification_baseline.model.dual_resnet import DualResNet

from master_thesis_benge_supervised_learning.classification_baseline.config.constants import Bands

# Import constants
from master_thesis_benge_supervised_learning.classification_baseline.config.constants import *

config = {
    "model": {
    # Model
        MODEL_KEY: DualResNet,
        MULTI_MODAL_KEY: True,
        WEIGHTS_KEY: False,
        NUMBER_OF_CLASSES_KEY: 11,
        NUMBER_OF_INPUT_CHANNELS_S1_KEY: 2,
        NUMBER_OF_INPUT_CHANNELS_S2_KEY: 12,
    },
    "training": {
        # Training
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
        # Data
        BANDS_KEY: Bands.ALL,
        TRANSFORMS_KEY: Transforms().transform,
        NORMALIZATION_VALUE_KEY: 10000,
        LABEL_THRESHOLD_KEY: 0.05,
        DATA_SET_SIZE_SMALL_KEY: True,
        SHUFFLE_TRAINING_DATA_KEY: True,
        SHUFFLE_VALIDATION_DATA_KEY: True,
    },
    "other": {
        CONFIG_NAME_KEY: "sentinel_2_sentinel_1",
        SAVE_MODEL_KEY: False,
        ENVIRONMENT_KEY: "remote",
    }
}