from segmentation.sen_12_floods import SEN12FLOODS
from torch.utils.data import DataLoader
from segmentation.model.unet import UNet
import torch
from train import Train

if __name__ == '__main__':

    trainset = SEN12FLOODS(
        root='../data/sen12floods/chips',
        transforms=True,
        split='train')

    valset = SEN12FLOODS(
        root='../data/sen12floods/chips',
        split='val')

    train_loader = DataLoader(
        trainset,
        batch_size=8,
        pin_memory=True)

    val_loader = DataLoader(
        valset,
        batch_size=8,
        pin_memory=True)

    #trainset.visualize_observation(8)

    model = UNet(13, 1)

    # Define a learning rate
    learning_rate = 0.01

    # Initialise the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Initialise the loss function and move it to the GPU if available
    criterion = torch.nn.BCEWithLogitsLoss()

    # First of all, let's verify if a GPU is available on our compute machine. If not, the cpu will be used instead.
    device = torch.device('cpu')

    train = Train(model, train_loader=train_loader, val_loader=val_loader, device=device, optimizer=optimizer, criterion=criterion).train()