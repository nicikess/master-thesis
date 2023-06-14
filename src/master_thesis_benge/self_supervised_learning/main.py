import wandb
import torch
import numpy as np

from master_thesis_benge.self_supervised_learning.config.constants import (
    TASK_KEY,
    SENTINEL_1_INDEX_KEY,
    CLIMATE_ZONE_INDEX_KEY,
    ERA_5_INDEX_KEY,
    SEASON_S2_INDEX_KEY,
    GLO_30_DEM_INDEX_KEY,
    SENTINEL_2_INDEX_KEY,
    ESA_WORLD_COVER_INDEX_KEY,
    TASK_CONFIG_KEY,
)

from master_thesis_benge.self_supervised_learning.training.train_setup import (
    train_setup
)

from master_thesis_benge.self_supervised_learning.model.model import (
    SimCLR_pl
)

from master_thesis_benge.supervised_baseline.config.config_runs.config_classification_landuse_multilabel import (
    training_config
)

from master_thesis_benge.self_supervised_learning.training.train import training

from master_thesis_benge.self_supervised_learning.evaluation.evaluation import evaluation


from ffcv.loader import Loader, OrderOption

if __name__ == "__main__":

    # Train SSL
    def train_setup():
        sweep_configuration = {
            "method": "grid",
            "name": "sweep",
            "parameters": {
                "batch_size": {"values": [128]},
                "temperature": {"values": [0.1]},
                "modalities": {"values": [
                                            [SENTINEL_1_INDEX_KEY, SENTINEL_2_INDEX_KEY]
                                        ]},
            },
        }

        sweep_id = wandb.sweep(
            sweep=sweep_configuration, project="self-supervised-test-run"
        )

        wandb.agent(sweep_id, function=training, count=20)


    def evaluation_setup():

        sweep_configuration = {
            "method": 'grid',
            "name": 'two-modality-ssl-test',
            "parameters": {
                "seed": {'values': [42, 43, 44, 45, 46]},
                #"learning_rate": {'values': [0.0001]},
                #"dataset_size": {'values': ["20"]},
                "modalities": {'values':    [
                                                [SENTINEL_1_INDEX_KEY, SENTINEL_2_INDEX_KEY],
                                            ]
                            },
            }
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project='master-thesis-self-supervised-'+training_config[TASK_CONFIG_KEY][TASK_KEY].lower())

        wandb.agent(sweep_id, function=evaluation)

    
    # Train
    train_setup()

    # Evaluate
    evaluation_setup()
