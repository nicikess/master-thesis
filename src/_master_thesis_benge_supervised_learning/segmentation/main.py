from master_thesis_benge_supervised_learning.segmentation.sen_12_floods import (
    SEN12FLOODS,
)
from torch.utils.data import DataLoader
from master_thesis_benge_supervised_learning.segmentation.model.unet import UNet
from train import Train
from master_thesis_benge_supervised_learning.segmentation.config.constants import (
    LocalFilesAndDirectoryReferences,
)
import wandb
from master_thesis_benge_supervised_learning.segmentation.config.config import *

from master_thesis_benge_supervised_learning.segmentation.dataset.ben_ge_s import BenGeS

if __name__ == "__main__":
    # Set references to files and directories
    esa_world_cover_data_train = pd.read_csv(
        LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_TRAIN
    )
    esa_world_cover_data_validation = pd.read_csv(
        LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_CSV_VALIDATION
    )
    sentinel_1_2_metadata = pd.read_csv(
        LocalFilesAndDirectoryReferences.SENTINEL_1_2_METADATA_CSV
    )
    era5_data = pd.read_csv(LocalFilesAndDirectoryReferences.ERA5_CSV)

    root_dir_s1 = LocalFilesAndDirectoryReferences.SENTINEL_1_DIRECTORY
    root_dir_s2 = LocalFilesAndDirectoryReferences.SENTINEL_2_DIRECTORY
    root_dir_world_cover = LocalFilesAndDirectoryReferences.ESA_WORLD_COVER_DIRECTORY

    # Create dataset
    dataset_train = BenGeS(
        esa_world_cover_data=esa_world_cover_data_train,
        sentinel_1_2_metadata=sentinel_1_2_metadata,
        era5_data=era5_data,
        root_dir_s1=root_dir_s1,
        root_dir_s2=root_dir_s2,
        root_dir_world_cover=root_dir_world_cover,
        wandb=wandb,
    )
    # wandb.log({"Dataset size": len(dataset_train)})

    dataset_validation = BenGeS(
        esa_world_cover_data=esa_world_cover_data_validation,
        sentinel_1_2_metadata=sentinel_1_2_metadata,
        era5_data=era5_data,
        root_dir_s1=root_dir_s1,
        root_dir_s2=root_dir_s2,
        root_dir_world_cover=root_dir_world_cover,
        wandb=wandb,
    )

    # Define training dataloader
    train_dl = DataLoader(
        dataset_train,
        config.get(batch_size_key),
        shuffle=config.get(shuffle_training_data_key),
    )

    # Define validation dataloader
    validation_dl = DataLoader(
        dataset_validation,
        config.get(batch_size_key),
        shuffle=config.get(shuffle_validation_data_key),
    )

    # trainset.visualize_observation(8)

    model = UNet(3, 11)

    # Define a learning rate
    learning_rate = 0.01

    # Initialise the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Initialise the loss function and move it to the GPU if available
    criterion = torch.nn.CrossEntropyLoss()

    # First of all, let's verify if a GPU is available on our compute machine. If not, the cpu will be used instead.
    device = torch.device("cpu")

    train = Train(
        model,
        train_loader=train_dl,
        val_loader=validation_dl,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
    ).train()
