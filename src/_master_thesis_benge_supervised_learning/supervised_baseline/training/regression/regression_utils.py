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

        '''
        self.epoch_train_rmse += self.

    def log_epoch_train_metrics(self, len_train_dataloader, scheduler):
        pass

    def reset_epoch_validation_metrics(self):
        pass

    def log_batch_validation_metrics(self, output, label):
        pass

    def log_epoch_validation_metrics(self, len_vali_dataloader):
        pass

    def calculate_and_log_accuracy_per_class_training(self, accuracy):
        pass

    def calculate_and_log_accuracy_per_class_validation(self, accuracy):
        pass

    def calculate_and_log_f1_per_class_training(self, accuracy):
        pass

    def calculate_and_log_f1_per_class_validation(self, accuracy):
        pass
        
        
        '''