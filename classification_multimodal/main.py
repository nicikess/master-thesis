import pandas as pd
import torch
from torch.utils.data import DataLoader
from ben_ge_s import BenGeS
from model.dual_resnet import DualResNet
#from model.resnet import ResNet
from transforms import Transforms
from train import HyperParameter
from train import Train
from display_example_image import DisplayExampleImage
import numpy as np
import wandb
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":

    environment = "remote"

    if environment == "local":
        #skiprows=lambda i: i % 15 != 0
        data_index = pd.read_csv("../data/ben-ge-s/ben-ge-s_esaworldcover.csv")#, skiprows=skiprows)
        root_dir = "../data/ben-ge-s/"
        device = torch.device("cpu")

    if environment == "remote":
        data_index = pd.read_csv("/ds2/remote_sensing/ben-ge/ben-ge-s/ben-ge-s_sentinel12_meta.csv")
        esaworldcover_index = pd.read_csv("/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy/ben-ge-s_esaworldcover.csv")      
        root_dir_s1 = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-1/s1_npy/"
        root_dir_s2 = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy/"
        device = torch.device("cuda")

    # Get transforms
    transforms = Transforms().transform

    # Define configurations
    torch.manual_seed(42)
    np.random.seed(42)
    model_description = input("Enter description for training run: ")
    config = {
        "model": DualResNet,
        "epochs": 20,
        "learning_rate": 0.001,
        "batch_size": 1,
        "optimizer": torch.optim.Adam,
        "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
        "loss": torch.nn.BCEWithLogitsLoss(),
        "t_max": 4_000,
        "eta_min": 1e-5,
        "bands": "RGB",
        "number_of_classes": 11,
        "number_of_input_channels": 3,
        "model_description": model_description,
    }

    wandb.login(key='9da448bfaa162b572403e1551114a17058f249d0')
    wandb.init(project="master-thesis", entity="nicikess", config=config)

    # Create dataset
    dataset = BenGeS(
        data_index,
        esaworldcover_index,
        root_dir_s1,
        root_dir_s2,
        number_of_classes=config.get("number_of_classes"),
        bands=config.get("bands"),
        transform=transforms,
    )
    # Random split
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    train_ds, validation_ds = data.random_split(
        dataset, [train_set_size, valid_set_size],
    )

    # Display example image
    # DisplayExampleImage(train_ds[0]).show_example_image()

    # Define training dataloader
    train_dl = DataLoader(train_ds, batch_size=config.get("batch_size"), shuffle=True)

    # Define validation dataloader
    validation_dl = DataLoader(validation_ds, batch_size=config.get("batch_size"), shuffle=False)

    # Set hyper parameters
    hyper_parameter = HyperParameter(
        epochs=config.get("epochs"),
        batch_size=config.get("batch_size"),
        learning_rate=config.get("learning_rate"),
        optimizer=config.get("optimizer"),
        scheduler=config.get("scheduler"),
        model_description=config.get("model_description"),
        loss=config.get("loss"),
        t_max=config.get("t_max"),
        eta_min=config.get("eta_min")
    )

    # Set number of classes and load model
    model = config.get("model")(
        in_channels_1=2,
        in_channels_2=3,
        number_of_classes=config.get("number_of_classes")
    )

    # Run training routing
    train = Train(
        model,
        train_dl=train_dl,
        validation_dl=validation_dl,
        device=device,
        wandb=wandb,
        hyper_parameter=hyper_parameter,
    ).train()