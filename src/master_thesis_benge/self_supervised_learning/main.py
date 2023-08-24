import wandb
import torch
import numpy as np
import os

from master_thesis_benge.self_supervised_learning.config.constants import (
    TASK_KEY,
    TRAINING_CONFIG_KEY,
    SENTINEL_1_INDEX_KEY,
    CLIMATE_ZONE_INDEX_KEY,
    ERA_5_INDEX_KEY,
    SEASON_S2_INDEX_KEY,
    GLO_30_DEM_INDEX_KEY,
    SENTINEL_2_INDEX_KEY,
    ESA_WORLD_COVER_INDEX_KEY,
    TASK_CONFIG_KEY,
    PIPELINES_CONFIG_KEY,
    TRAINING_RESNET_CONFIG_KEY,
    TRAINING_UNET_CONFIG_KEY,
    EVALUATION_CLASSIFICATION_LANDUSE_MULTILABEL_CONFIG_KEY,
    EVALUATION_SEGMENTATION_LANDUSE_CONFIG_KEY,
    EVALUATION_REGRESSION_LANDUSE_FRACTION_CONFIG_KEY
)

from master_thesis_benge.self_supervised_learning.training.train import training

from master_thesis_benge.self_supervised_learning.evaluation.evaluation import evaluation

from ffcv.loader import Loader, OrderOption

if __name__ == "__main__":

    # Train SSL
    def train_setup():

        training_config = TRAINING_RESNET_CONFIG_KEY

        sweep_configuration = {
            "method": "grid",
            "name": "train-ssl-sen2-sen1-glo-different-dataset-sizes-resnet",
            "parameters": {
                "training_config": {"values": [training_config]},
                "batch_size": {"values": [128]},
                "temperature": {"values": [0.1]},
                "dataset_size_train": {'values': [
                                                    #"60-delta-multilabel"
                                                    #"delta-60-40-s",
                                                    #"delta-80-40",
                                                    #"100",
                                                    "delta-100-20",
                                                    #"delta-100-40",
                                                ]},
                "modalities": {'values':    [
                                                [SENTINEL_2_INDEX_KEY, SENTINEL_1_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                                                #[SENTINEL_2_INDEX_KEY, ESA_WORLD_COVER_INDEX_KEY],
                                                #[SENTINEL_2_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                                                #[SENTINEL_1_INDEX_KEY, ESA_WORLD_COVER_INDEX_KEY],
                                                #[SENTINEL_1_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                                                #[ESA_WORLD_COVER_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                                            ]
                            },
            },
        }

        if training_config == TRAINING_RESNET_CONFIG_KEY:
            project_name = "master-thesis-ssl-training-resnet"
        elif training_config == TRAINING_UNET_CONFIG_KEY:
            project_name = "master-thesis-ssl-training-unet"

        sweep_id = wandb.sweep(
            sweep=sweep_configuration, project=project_name
        )

        wandb.agent(sweep_id, function=training)


    def evaluation_setup():

        def extract_ckpt_name(file_path):
            file_name = os.path.basename(file_path)
            ckpt_name = os.path.splitext(file_name)[0]
            return ckpt_name
        

        pre_trained_weights =   [   
                                    # ResNet
                                    #'saved_models/resnet_weights/i135szoj/sentinel2-sentinel1-delta-100-20.ckpt',
                                    #'saved_models/resnet_weights/i135szoj/sentinel2-sentinel1-delta-100-40.ckpt',
                                    #'saved_models/resnet_weights/s4vzi5bq/sentinel2-sentinel1-delta-80-40.ckpt',
                                    #'saved_models/resnet_weights/s4vzi5bq/sentinel2-sentinel1-60-delta-multilabel.ckpt',
                                    #'saved_models/resnet_weights/s4vzi5bq/sentinel2-sentinel1-delta-60-40-s.ckpt'

                                    # ResNet triple weights
                                    'saved_models/resnet_weights/nkzdt0o0/sentinel2-sentinel1-elevation(glo-30-dem)-delta-100-20.ckpt',

                                    # UNet
                                    #'saved_models/unet_weights/v1l0llek/sentinel2-sentinel1-delta-100-20.ckpt',
                                ]
    
        modalities =            [
                                    #[SENTINEL_1_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                                    #[SENTINEL_1_INDEX_KEY, ESA_WORLD_COVER_INDEX_KEY],
                                    #[SENTINEL_2_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                                    [SENTINEL_2_INDEX_KEY, SENTINEL_1_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                                    #[SENTINEL_2_INDEX_KEY, SENTINEL_1_INDEX_KEY],
                                    #[SENTINEL_2_INDEX_KEY, ESA_WORLD_COVER_INDEX_KEY],
                                    #[ESA_WORLD_COVER_INDEX_KEY, GLO_30_DEM_INDEX_KEY],
                                ]
        
        sweep_name =            [
                                    #'eval-ssl-sen2-sen1-1-percent',
                                    #'eval-ssl-sen2-sen1-10-percent',
                                    #'eval-ssl-sen2-sen1-50-percent',
                                    #'eval-ssl-sen2-sen1-100-percent',
                                ]
        
        evaluation_task = EVALUATION_CLASSIFICATION_LANDUSE_MULTILABEL_CONFIG_KEY
        
        for i in range(len(pre_trained_weights)):
            sweep_configuration = {
                "method": 'grid',
                "name": extract_ckpt_name(pre_trained_weights[i]),
                "parameters": {
                    "evaluation_config": {'values': [evaluation_task]},
                    "seed": {'values': [42, 43, 44, 45, 46]},
                    "batch_size": {"values": [128]}, # only to init the SimCLR model
                    "temperature": {"values": [0.1]},  # only to init the SimCLR model
                    "pre_trained_weights_path": {'values': [pre_trained_weights[i]]},
                    "dataset_size_fine_tuning": {'values': [
                                                        "20-multi-label-ewc",
                                                        "20-50-percent",
                                                        "20-10-percent",
                                                        "20-1-percent",
                                                        ]},
                    "modalities": {'values':    [modalities[i]]
                                },
                }
            }

            if evaluation_task == EVALUATION_CLASSIFICATION_LANDUSE_MULTILABEL_CONFIG_KEY:
                project_name = 'master-thesis-ssl-evaluation-classification-landuse-multilabel'
            if evaluation_task == EVALUATION_SEGMENTATION_LANDUSE_CONFIG_KEY:
                project_name = 'master-thesis-ssl-evaluation-segmentation-landuse'
            elif evaluation_task == EVALUATION_REGRESSION_LANDUSE_FRACTION_CONFIG_KEY:
                project_name = 'master-thesis-ssl-evaluation-regression-landuse-fraction'

            sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
            wandb.agent(sweep_id, function=evaluation)

    
    # Train
    #train_setup()

    # Evaluate
    evaluation_setup()