import wandb

from master_thesis_benge.supervised_baseline.training.metric import Metric

from torchmetrics.regression import (
    MeanSquaredError
)

import torch.nn.functional as F

from master_thesis_benge.supervised_baseline.config.constants import (
    Task,
)

class RegressionUtils(Metric):
    # Train values
    epoch_train_loss = 0
    epoch_train_mse = 0
    epoch_train_rmse = 0
    # Per class
    epoch_train_mse_per_class = 0
    epoch_train_rmse_per_class = 0

    # Validation values
    epoch_val_mse = 0
    epoch_val_rmse = 0
    # Per class
    epoch_train_mse_per_class = 0
    epoch_train_rmse_per_class = 0

    def __init__(self, wandb, device, number_of_classes, task):
        self.wandb = wandb
        self.device = device
        self.number_of_classes = number_of_classes
        self.task = task

        self.mse = MeanSquaredError(squared=True).to(self.device)
        self.rsme = MeanSquaredError(squared=False).to(self.device)

        self.epoch_train_mse_per_class = [0.0] * self.number_of_classes
        self.epoch_train_rmse_per_class = [0.0] * self.number_of_classes
        self.epoch_validation_mse_per_class = [0.0] * self.number_of_classes
        self.epoch_validation_rmse_per_class = [0.0] * self.number_of_classes

    def calculate_loss(self, loss, output, label):
        if self.task == Task.REGRESSION_LANDUSE_FRACTION.value:
            output = F.softmax(output, dim=1)
        if self.task == Task.REGRESSION_ELEVATION_DIFFERENCE.value:
            output = F.sigmoid(output)
        if self.task == Task.SEGMENTATION_ELEVATION.value:
            output = F.sigmoid(output)
        loss = loss(output, label)
        return loss

    def reset_epoch_train_metrics(self):
        self.epoch_train_loss = 0
        self.epoch_train_mse = 0
        self.epoch_train_rmse = 0
        self.epoch_train_mse_per_class = [0.0] * self.number_of_classes
        self.epoch_train_rmse_per_class = [0.0] * self.number_of_classes

    def log_batch_train_metrics(self, loss, output, label, progress, epoch):
        self.epoch_train_loss += loss

        if self.task == Task.REGRESSION_LANDUSE_FRACTION.value:
            output = F.softmax(output, dim=1)
        if self.task == Task.REGRESSION_ELEVATION_DIFFERENCE.value:
            output = F.sigmoid(output)
        if self.task == Task.SEGMENTATION_ELEVATION.value:
            output = F.sigmoid(output)

        self.epoch_train_mse += self.mse(output, label)
        self.epoch_train_rmse += self.rsme(output, label)

        # Calculate MSE and RMSE for each class
        for class_id in range(self.number_of_classes):
            class_output = output[:, class_id]  # Extract the predictions for the current class
            class_label = label[:, class_id]    # Extract the ground truth for the current class

            class_mse = self.mse(class_output, class_label)
            class_rmse = self.rsme(class_output, class_label)

            # Add the class-specific MSE and RMSE to the corresponding epoch metrics
            self.epoch_train_mse_per_class[class_id] += class_mse
            self.epoch_train_rmse_per_class[class_id] += class_rmse

        progress.set_description("Train loss "+str(epoch)+":{:.4f}".format(loss))
        wandb.log({"Step loss": loss})

    def log_epoch_train_metrics(self, len_train_dataloader, scheduler):
        # Calculate average per metric per epoch
        epoch_train_loss = self.epoch_train_loss / len_train_dataloader
        epoch_train_mse = self.epoch_train_mse / len_train_dataloader
        epoch_train_rmse = self.epoch_train_rmse / len_train_dataloader
        epoch_train_mse_per_class = [mse / len_train_dataloader for mse in self.epoch_train_mse_per_class]
        epoch_train_rmse_per_class = [rmse / len_train_dataloader for rmse in self.epoch_train_rmse_per_class]

        print(f"\n epoch train loss: {epoch_train_loss} \n")

        wandb.log({"Epoch train loss": epoch_train_loss})
        wandb.log({"Epoch train mse": epoch_train_mse})
        wandb.log({"Epoch train rmse": epoch_train_rmse})
        wandb.log({"Learning-rate": scheduler.get_last_lr()[0]})

        # Log class-wise MSE and RMSE for train
        for class_id in range(self.number_of_classes):
            wandb.log({f"Epoch train mse (class {class_id})": epoch_train_mse_per_class[class_id]})
            wandb.log({f"Epoch train rmse (class {class_id})": epoch_train_rmse_per_class[class_id]})

    def reset_epoch_validation_metrics(self):
        self.epoch_val_mse = 0
        self.epoch_val_rmse = 0
        self.epoch_validation_mse_per_class = [0.0] * self.number_of_classes
        self.epoch_validation_rmse_per_class = [0.0] * self.number_of_classes

    def log_batch_validation_metrics(self, output, label):

        if self.task == Task.REGRESSION_LANDUSE_FRACTION.value:
            output = F.softmax(output, dim=1)
        if self.task == Task.REGRESSION_ELEVATION_DIFFERENCE.value:
            output = F.sigmoid(output)
        if self.task == Task.SEGMENTATION_ELEVATION.value:
            output = F.sigmoid(output)

        self.epoch_val_mse += self.mse(output, label)
        self.epoch_val_rmse += self.rsme(output, label)

        # Calculate MSE and RMSE for each class
        for class_id in range(self.number_of_classes):
            class_output = output[:, class_id]  # Extract the predictions for the current class
            class_label = label[:, class_id]    # Extract the ground truth for the current class

            class_mse = self.mse(class_output, class_label)
            class_rmse = self.rsme(class_output, class_label)

            # Add the class-specific MSE and RMSE to the corresponding epoch metrics
            self.epoch_validation_mse_per_class[class_id] += class_mse
            self.epoch_validation_rmse_per_class[class_id] += class_rmse

    def log_epoch_validation_metrics(self, len_vali_dataloader):
        # Calculate average per metric per epoch
        epoch_val_mse = self.epoch_val_mse / len_vali_dataloader
        epoch_val_rmse = self.epoch_val_rmse / len_vali_dataloader
        epoch_val_mse_per_class = [mse / len_vali_dataloader for mse in self.epoch_validation_mse_per_class]
        epoch_val_rmse_per_class = [rmse / len_vali_dataloader for rmse in self.epoch_validation_rmse_per_class]

        wandb.log({"Epoch val mse": epoch_val_mse})
        wandb.log({"Epoch val rmse": epoch_val_rmse})

        # Log class-wise MSE and RMSE for validation
        for class_id in range(self.number_of_classes):
            wandb.log({f"Epoch val mse (class {class_id})": epoch_val_mse_per_class[class_id]})
            wandb.log({f"Epoch val rmse (class {class_id})": epoch_val_rmse_per_class[class_id]})
