import wandb

from _master_thesis_benge_supervised_learning.supervised_baseline.training.metric import Metric

from torchmetrics.regression import (
    MeanSquaredError
)

from torch import nn

class RegressionUtils(Metric):
    # Train values
    epoch_train_loss = 0
    epoch_train_mse = 0
    epoch_train_rmse = 0

    # Validation values
    epoch_val_mse = 0
    epoch_val_rmse = 0

    def __init__(self, wandb, device, number_of_classes):
        self.wandb = wandb
        self.device = device
        self.number_of_classes = number_of_classes

        self.mse = MeanSquaredError(squared=False)
        self.rsme = MeanSquaredError(squared=True)


    def reset_epoch_train_metrics(self):
        self.epoch_train_loss = 0
        self.epoch_train_mse = 0
        self.epoch_train_rmse = 0

    def log_batch_train_metrics(self, loss, output, label, progress):
        self.epoch_train_loss += loss

        sigmoid = nn.Sigmoid()
        sigmoid_output = sigmoid(output)

        self.epoch_train_mse += self.mse(sigmoid_output, label)
        self.epoch_train_rmse += self.rsme(sigmoid_output, label)

        progress.set_description("Train loss epoch: {:.4f}".format(loss))
        wandb.log({"Step loss": loss})

    def log_epoch_train_metrics(self, len_train_dataloader, scheduler):
        # Calculate average per metric per epoch
        epoch_train_loss = self.epoch_train_loss / len_train_dataloader
        epoch_train_mse = self.epoch_train_mse / len_train_dataloader
        epoch_train_rmse = self.epoch_train_rmse / len_train_dataloader

        print(f"\n epoch train loss: {epoch_train_loss} \n")

        wandb.log({"Epoch train loss": epoch_train_loss})
        wandb.log({"Epoch train mse": epoch_train_mse})
        wandb.log({"Epoch train rmse": epoch_train_rmse})
        wandb.log({"Learning-rate": scheduler.get_last_lr()[0]})

    def reset_epoch_validation_metrics(self):
        self.epoch_val_mse = 0
        self.epoch_val_rmse = 0

    def log_batch_validation_metrics(self, output, label):
        sigmoid = nn.Sigmoid()
        sigmoid_output = sigmoid(output)

        self.epoch_val_mse += self.mse(sigmoid_output, label)
        self.epoch_val_rmse += self.rsme(sigmoid_output, label)

    def log_epoch_validation_metrics(self, len_vali_dataloader):
        # Calculate average per metric per epoch
        epoch_val_mse = self.epoch_train_mse / len_vali_dataloader
        epoch_val_rmse = self.epoch_train_rmse / len_vali_dataloader

        wandb.log({"Epoch val mse": epoch_val_mse})
        wandb.log({"Epoch val rmse": epoch_val_rmse})