from enum import Enum

import pandas as pd
import torch
import torchvision.models as models

from master_thesis_benge_supervised_learning.classification_baseline.config.constants import Bands

model_key = "model"
model_size_key = "model_size"
multi_modal_key = "multi_modal"
weights_key = "weights"
epochs_key = "epochs"
learning_rate_key = "learning_rate"
batch_size_key = "batch_size"
optimizer_key = "optimizer"
scheduler_key = "scheduler"
loss_key = "loss"
bands_key = "bands"
number_of_classes_key = "number_of_classes"
number_of_input_channels_s1_key = "number_of_input_channels_s1"
number_of_input_channels_s2_key = "number_of_input_channels_s2"
seed_key = "seed"
transforms_key = "transforms"
normalization_value_key = "normalization_value"
label_threshold_key = "label_threshold"
scheduler_max_number_interations_key = "scheduler_max_number_interations"
scheduler_min_lr_key = "scheduler_min_lr"
save_model_key = "save_model"
shuffle_training_data_key = "shuffle_training_data"
shuffle_validation_data_key = "shuffle_validation_data"
data_set_size_small_key = "data_set_size_small"
environment_key = "environment"

config = {
    # Model
    #model_key: ResNet,
    multi_modal_key: False,
    weights_key: False,
    number_of_classes_key: 11,
    number_of_input_channels_s1_key: 2,
    number_of_input_channels_s2_key: 3,
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
    # Data
    bands_key: Bands.RGB,
    #transforms_key: Transforms().transform,
    normalization_value_key: 10000,
    label_threshold_key: 0.3,
    data_set_size_small_key: True,
    shuffle_training_data_key: True,
    shuffle_validation_data_key: True,
    # Other
    save_model_key: False,
    environment_key: "local",
}

config_recommendation = {
    "model": {
    # Model
        #model_key: ResNet,
        multi_modal_key: False,
        weights_key: False,
        number_of_classes_key: 11,
        number_of_input_channels_s1_key: 2,
        number_of_input_channels_s2_key: 3,
    },

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
    "data": {
        # Data
        bands_key: Bands.RGB,
        #transforms_key: Transforms().transform,
        normalization_value_key: 10000,
        label_threshold_key: 0.3,
        data_set_size_small_key: True,
        shuffle_training_data_key: True,
        shuffle_validation_data_key: True,
    },
    # Other
    save_model_key: False,
    environment_key: "local",
}


if __name__ == '__main__':
    from master_thesis_benge_supervised_learning.classification_baseline.dataset.ben_ge_s import BenGeS
    data = BenGeS(
        esa_world_cover_data=pd.DataFrame({"0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}),
        **config_recommendation["data"],
    )