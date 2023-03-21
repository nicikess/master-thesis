import pandas as pd
import torch
from torch.utils.data import DataLoader
from ben_ge_s import BenGeS
from model.resnet import ResNet
from transforms import Transforms
from train import HyperParameter
from train import Train
import numpy as np
import wandb
import torch.utils.data as data

if __name__ == "__main__":

    environment = "local"

    if environment == "local":
        data_index = pd.read_csv("../data/ben-ge-s/ben-ge-s_esaworldcover.csv")
        root_dir = "../data/ben-ge-s/"
        device = torch.device("cpu")

    if environment == "remote":
        data_index = pd.read_csv("/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy/ben-ge-s_esaworldcover.csv")
        root_dir = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy/"
        device = torch.device("cuda")

    # Get transforms
    transforms = Transforms().transform

    # Model description for documentation
    model_description = input("Enter description for training run: ")

    config = {
        # Training
        "epochs": 20,
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": torch.optim.Adam,
        "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
        "loss": torch.nn.BCEWithLogitsLoss(),
        "bands": "RGB",
        "number_of_classes": 11,
        "number_of_input_channels": 3,
        "model_description": model_description,
        # Other
        "seed": 42
    }

    # sort numpy array


    # Define configurations
    torch.manual_seed(config.get("seed"))
    np.random.seed(config.get("seed"))

    #if environment == "remote":
    #wandb.login(key='9da448bfaa162b572403e1551114a17058f249d0')
    #wandb.init(project="master-thesis", entity="nicikess", config=config)

    # Create dataset
    dataset = BenGeS(
        data_index,
        root_dir,
        wandb=wandb,
        number_of_classes=config.get("number_of_classes"),
        bands=config.get("bands"),
        transform=transforms
    )
    # Random split
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    train_ds, validation_ds = data.random_split(
        dataset, [train_set_size, valid_set_size],
    )
    #wandb.log({"Length dataset": len(dataset)})

    #Leave it!
    '''
    count = 0
    for img in train_ds:
        img = img[1]
        img = img[img>10_000]
        print(len(img))
        #if len(img > 0):
            #count += 1
    print(count)
    '''

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
        loss=config.get("loss")
    )

    # Set number of classes and load model
    model = ResNet(
        number_of_input_channels=config.get("number_of_input_channels"),
        number_of_classes=config.get("number_of_classes"),
    ).model
    #wandb.log({"Model": model})

    # Run training routing
    train = Train(
        model,
        train_dl=train_dl,
        validation_dl=validation_dl,
        number_of_classes=config.get("number_of_classes"),
        device=device,
        wandb=wandb,
        hyper_parameter=hyper_parameter,
        environment=environment
    ).train()