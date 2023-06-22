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

from master_thesis_benge.self_supervised_learning.model.sim_clr_model import (
    SimCLR_pl
)

from master_thesis_benge.self_supervised_learning.config.config_self_supervised_learning_evaluation_classification_landuse_multilabel import (
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
            "name": "dataset-size-evaluation-ssl",
            "parameters": {
                "batch_size": {"values": [128]},
                "temperature": {"values": [0.1]},
                "dataset_size": {'values': ["8k","20","40", "60", "80", "100"]},
                "modalities": {'values':    [
                                                [SENTINEL_2_INDEX_KEY, SENTINEL_1_INDEX_KEY],
                                            ]
                            },
            },
        }

        sweep_id = wandb.sweep(
            sweep=sweep_configuration, project="master-thesis-ssl-training"
        )

        wandb.agent(sweep_id, function=training)


    def evaluation_setup():

        sweep_configuration = {
            "method": 'grid',
            "name": 'two-modality',
            "parameters": {
                "seed": {'values': [42]},
                #"learning_rate": {'values': [0.0001]},
                #"dataset_size": {'values': ["20"]},
                "batch_size": {"values": [128]}, # only to init the SimCLR model
                "temperature": {"values": [0.1]},  # only to init the SimCLR model
                "pre_trained_weights_path": {'values': [#'saved_models/sentinel2-sentinel1-modality-run-6g8bi57r.ckpt',
                                                        #'saved_models/worldcover(esa)-sentinel1.ckpt',
                                                        #'saved_models/worldcover(esa)-sentinel2.ckpt',
                                                        #'saved_models/worldcover(esa)-elevation(glo-30-dem).ckpt',
                                                        #'saved_models/sentinel2-elevation(glo-30-dem).ckpt',
                                                        'saved_models/sentinel1-elevation(glo-30-dem).ckpt',
                                                        ]
                                                        },
                "modalities": {'values':    [
                                                #[SENTINEL_2_INDEX_KEY, SENTINEL_1_INDEX_KEY],
                                                #[ESA_WORLD_COVER_INDEX_KEY, SENTINEL_1_INDEX_KEY],
                                                #[ESA_WORLD_COVER_INDEX_KEY, SENTINEL_2_INDEX_KEY],
                                                #[ESA_WORLD_COVER_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                                                #[SENTINEL_2_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                                                [SENTINEL_1_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                                            ]
                            },
            }
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project='master-thesis-ssl-evaluation-'+training_config[TASK_CONFIG_KEY][TASK_KEY].lower())

        wandb.agent(sweep_id, function=evaluation)

    
    # Train
    #train_setup()

    # Evaluate
    evaluation_setup()