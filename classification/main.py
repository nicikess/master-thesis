import pandas as pd
import torch
from torch.utils.data import DataLoader
from ben_ge_s import BenGeS
from model.dual_resnet import DualResNet
from model.resnet import ResNet
from transforms import Transforms
from train import HyperParameter
from train import Train
from display_example_image import DisplayExampleImage
import numpy as np
import wandb
import torch.utils.data as data

if __name__ == '__main__':
    # Load index file
    data_index = pd.read_csv('data/ben-ge-s/ben-ge-s_esaworldcover.csv')

    # Set root dir
    root_dir = 'data/ben-ge-s/'

    # Get transforms
    transforms = Transforms().transform

    # Define configurations
    torch.manual_seed(42)
    model_description = input("Enter description for training run: ")
    config = {
        "epochs": 20,
        "learning_rate": 0.001,
        "batch_size": 32,
        "opt_func": torch.optim.Adam,
        "milestones": [5, 15],
        "weight_decay": 0,
        "loss": torch.nn.CrossEntropyLoss(),
        "number_of_classes": 11,
        "number_of_input_channels": 3,
        "model_description": model_description
    }

    wandb.login(key='9da448bfaa162b572403e1551114a17058f249d0')
    wandb.init(project="master-thesis", entity="nicikess")

    # Create dataset
    dataset = BenGeS(data_index, root_dir, number_of_classes=config.get("number_of_classes"), bands_rgb=True, transform=transforms)
    # Random split
    train_set_size = int(len(dataset) * 0.08)
    valid_set_size = len(dataset) - train_set_size
    train_ds, validation_ds = data.random_split(dataset, [train_set_size, valid_set_size],)

    # Display example image
    # DisplayExampleImage(train_ds[0]).show_example_image()

    # Define training dataloader
    train_dl = DataLoader(train_ds, batch_size=config.get("batch_size"), shuffle=False)

    # Define validation dataloader
    validation_dl = DataLoader(validation_ds, batch_size=config.get("batch_size"), shuffle=False)

    # Set hyperparameters
    hyper_parameter = HyperParameter(epoch=config.get("epochs"),
                                     batch_size=config.get("batch_size"),
                                     learning_rate=config.get("learning_rate"),
                                     opt_func=config.get("opt_func"),
                                     milestones=config.get("milestones"),
                                     weight_decay=config.get("weight_decay"),
                                     model_description=config.get("model_description"),
                                     loss=config.get("loss"))

    # Set number of classes and load model
    model = ResNet(number_of_input_channels=config.get("number_of_input_channels"), number_of_classes=config.get("number_of_classes")).model

    # Run training routing
    device = torch.device('cpu')
    print(type(device.type))
    train = Train(model, train_dl=train_dl, validation_dl=validation_dl, device=device, wandb=wandb, hyper_parameter=hyper_parameter).train()
