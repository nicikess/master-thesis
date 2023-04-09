from enum import Enum

import pandas as pd
import torch
import torchvision.models as models

# Import transforms
from master_thesis_benge_supervised_learning.classification_baseline.training.transforms import Transforms

# Import models
from master_thesis_benge_supervised_learning.classification_baseline.model.dual_resnet import DualResNet

from master_thesis_benge_supervised_learning.classification_baseline.config.constants import Bands

config = {
    "model": {
    # Model
        model_key: DualResNet,
        multi_modal_key: True,
        weights_key: False,
        number_of_classes_key: 11,
        number_of_input_channels_s1_key: 2,
        number_of_input_channels_s2_key: 12,
    },
    "training": {
        # Training
        epochs_key: 20,
        learning_rate_key: 0.001,
        batch_size_key: 32,
        optimizer_key: torch.optim.Adam,
        scheduler_key: torch.optim.lr_scheduler.CosineAnnealingLR,
        loss_key: torch.nn.BCEWithLogitsLoss(),
        seed_key: 42,
        scheduler_max_number_interations_key: 20,
        scheduler_min_lr_key: 0,
    },
    "data": {
        # Data
        bands_key: Bands.ALL,
        transforms_key: Transforms().transform,
        normalization_value_key: 10000,
        label_threshold_key: 0.05,
        data_set_size_small_key: True,
        shuffle_training_data_key: True,
        shuffle_validation_data_key: True,
    },
    "other": {
        save_model_key: False,
        environment_key: "remote",
    }
}

'''
if __name__ == '__main__':
    from master_thesis_benge_supervised_learning.classification_baseline.dataset.ben_ge_s import BenGeS
    data = BenGeS(
        esa_world_cover_data=pd.DataFrame({"0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}),
        **config_recommendation["data"],
    )
'''