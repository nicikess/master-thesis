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

# TODO: Change this for training: training_config_resnet or training_config_unet
from master_thesis_benge.self_supervised_learning.config.config_self_supervised_learning_training import training_config_resnet as training_config

# TODO: Change task for evaluation
from master_thesis_benge.self_supervised_learning.config.config_self_supervised_learning_evaluation_classification_landuse_multilabel import (
    evaluation_config
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
                "training_config": {"values": [training_config]},
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
            sweep=sweep_configuration, project="master-thesis-ssl-training"+training_config[TASK_CONFIG_KEY][TASK_KEY]
        )

        wandb.agent(sweep_id, function=training)


    def evaluation_setup():

        pre_trained_weights = [
                    'saved_models/sentinel2-sentinel1-modality-run-6g8bi57r.ckpt',
                    'saved_models/worldcover(esa)-sentinel1.ckpt',
                    'saved_models/worldcover(esa)-sentinel2.ckpt',
                    'saved_models/worldcover(esa)-elevation(glo-30-dem).ckpt',
                    'saved_models/sentinel2-elevation(glo-30-dem).ckpt',
                    'saved_models/sentinel1-elevation(glo-30-dem).ckpt',
                        ]
    
        modalities = [
                        [SENTINEL_2_INDEX_KEY, SENTINEL_1_INDEX_KEY],
                        [ESA_WORLD_COVER_INDEX_KEY, SENTINEL_1_INDEX_KEY],
                        [ESA_WORLD_COVER_INDEX_KEY, SENTINEL_2_INDEX_KEY],
                        [ESA_WORLD_COVER_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                        [SENTINEL_2_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                        [SENTINEL_1_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                                                ]
        
        sweep_name = [
                    'sentinel2-sentinel1-modality',
                    'worldcover(esa)-sentinel1',
                    'worldcover(esa)-sentinel2',
                    'worldcover(esa)-elevation(glo-30-dem)',
                    'sentinel2-elevation(glo-30-dem)',
                    'sentinel1-elevation(glo-30-dem)'
                    ]
        
        for i in range(len(pre_trained_weights)):
            sweep_configuration = {
                "method": 'grid',
                "name": sweep_name[i],
                "parameters": {
                    "evaluation_config": {'values': [evaluation_config]},
                    "seed": {'values': [42]},
                    "batch_size": {"values": [128]}, # only to init the SimCLR model
                    "temperature": {"values": [0.1]},  # only to init the SimCLR model
                    "pre_trained_weights_path": {'values': [pre_trained_weights[i]]},
                    "modalities": {'values':    [modalities[i]]
                                },
                }
            }

            sweep_id = wandb.sweep(sweep=sweep_configuration, project='master-thesis-ssl-evaluation-'+evaluation_config[TASK_CONFIG_KEY][TASK_KEY].lower())
            wandb.agent(sweep_id, function=evaluation)

    
    # Train
    #train_setup()

    # Evaluate
    evaluation_setup()