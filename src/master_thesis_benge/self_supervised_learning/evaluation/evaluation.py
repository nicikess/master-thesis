import wandb
import torch
import numpy as np

from master_thesis_benge.self_supervised_learning.config.constants import (
    MODEL_KEY,
    TRAINING_CONFIG_KEY,
    BATCH_SIZE_KEY,
    NUMBER_OF_CLASSES_KEY,
    METRICS_KEY,
    TASK_KEY,
    DATALOADER_TRAIN_FILE_KEY,
    DATALOADER_VALIDATION_FILE_KEY,
    PIPELINES_CONFIG_KEY,
    TASK_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    METRICS_CONFIG_KEY,
    FEATURE_DIMENSION_KEY,
    EVALUATION_CLASSIFICATION_LANDUSE_MULTILABEL_CONFIG_KEY,
    EVALUATION_SEGMENTATION_LANDUSE_CONFIG_KEY,
    get_label_from_index
)

from master_thesis_benge.supervised_baseline.training.train import (
    Train
)

from master_thesis_benge.self_supervised_learning.model.training.sim_clr_model import (
    SimCLR_pl
)

from master_thesis_benge.self_supervised_learning.config.config_self_supervised_learning_evaluation_classification_landuse_multilabel import (
    evaluation_config_classification_landuse_multilabel, get_data_set_files
)

from master_thesis_benge.self_supervised_learning.config.config_self_supervised_learning_evaluation_segmentaion_landuse import (
    evaluation_config_segmenation_landuse
)

from ffcv.loader import Loader, OrderOption

def select_evaluation_config(config_type):
    if config_type == EVALUATION_CLASSIFICATION_LANDUSE_MULTILABEL_CONFIG_KEY:
        return evaluation_config_classification_landuse_multilabel
    elif config_type == EVALUATION_SEGMENTATION_LANDUSE_CONFIG_KEY:
        return evaluation_config_segmenation_landuse
    else:
        raise ValueError("Invalid config type.")

def evaluation():

    wandb.init()
    evaluation_config = select_evaluation_config(wandb.config.evaluation_config)
    wandb.config.update(evaluation_config)

    run_name = '-'.join([get_label_from_index(modality) for modality in wandb.config.modalities])
    wandb.run.name = run_name

    # Set device
    device = torch.device('cuda')

    # Set seed
    torch.manual_seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)

    #get_data_set_files(wandb.config.dataset_size)[0]
    dataloader_train = Loader(get_data_set_files(wandb.config.dataset_size_fine_tuning)[0],
                            batch_size=evaluation_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
                            order=OrderOption.RANDOM,
                            num_workers=4,
                            pipelines=evaluation_config[PIPELINES_CONFIG_KEY]
                        )

    dataloader_validation = Loader(evaluation_config[TRAINING_CONFIG_KEY][DATALOADER_VALIDATION_FILE_KEY],
                            batch_size=evaluation_config[TRAINING_CONFIG_KEY][BATCH_SIZE_KEY],
                            order=OrderOption.RANDOM,
                            num_workers=4,
                            pipelines=evaluation_config[PIPELINES_CONFIG_KEY]
                        )
        
    # Create a dictionary that maps each modality to the number of input channels
    channel_modalities = {
        f"in_channels_{i+1}": int(str(np.shape(next(iter(dataloader_train))[modality])[1]))
        for i, modality in enumerate(
            wandb.config.modalities
        )
    }

    '''
    itera = iter(dataloader_train)
    first = next(itera)
    for data in first:
        print(data)
        print(data.shape)
        input("test")
    '''

    # Load weights from pre-trained model
    model_ssl = SimCLR_pl.load_from_checkpoint(wandb.config.pre_trained_weights_path, training_config=evaluation_config, in_channels_1=channel_modalities["in_channels_1"], in_channels_2=channel_modalities["in_channels_2"])

    state_dict_modality_1 = model_ssl.model_modality_1.backbone.state_dict()
    state_dict_modality_2 = model_ssl.model_modality_2.backbone.state_dict()

    state_dict = {
        "state_dict_modality_1": state_dict_modality_1,
        "state_dict_modality_2": state_dict_modality_2
    }

    # Define model
    model_sl = evaluation_config[MODEL_CONFIG_KEY][MODEL_KEY](
        # Define multi modal model
        # Input channels for s1
        state_dict=state_dict,
        in_channels_1=channel_modalities["in_channels_1"],
        #in_channels_1=4,
        # Input channels for s2
        in_channels_2=channel_modalities["in_channels_2"],
        #in_channels_3=channel_modalities["in_channels_3"],
        number_of_classes=evaluation_config[MODEL_CONFIG_KEY][NUMBER_OF_CLASSES_KEY],
    )

    wandb.log({"model details": model_sl})
    wandb.log({"Notes": f'Modalities: {wandb.config.modalities} with data set train size: {len(dataloader_train)}'})

    # Run training routing
    train = Train(
        model_sl,
        train_dl=dataloader_train,
        validation_dl=dataloader_validation,
        metrics=evaluation_config[METRICS_CONFIG_KEY][METRICS_KEY],
        wandb=wandb,
        device=device,
        config=evaluation_config,
        task=evaluation_config[TASK_CONFIG_KEY][TASK_KEY],
        modalities=wandb.config.modalities
    ).train()