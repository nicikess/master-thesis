import wandb

from torchmetrics import (
    JaccardIndex
)

from torch import nn
import torch

from master_thesis_benge.supervised_baseline.training.metric import Metric

import torch.nn.functional as F

class SegmentationUtils(Metric):

    # Train values
    epoch_train_loss = 0
    epoch_train_accuracy = 0
    #epoch_train_accuracy_per_class = 0
    #epoch_train_f1_score
    epoch_train_jaccard = 0

    # Validation values
    epoch_validation_jaccard = 0
    epoch_validation_accuracy = 0

    def __init__(self, wandb, device, number_of_classes, task):
        self.wandb = wandb
        self.device = device,
        self.number_of_classes = number_of_classes
        self.task = task
        self.jaccard = JaccardIndex('multiclass', num_classes=self.number_of_classes).to(device=torch.device('cuda'))

    def calculate_loss(self, loss, output, label):
        loss = loss(output, label)
        return loss

    def reset_epoch_train_metrics(self):
        self.epoch_train_loss = 0
        self.epoch_train_jaccard = 0
        self.epoch_train_accuracy = 0

    def log_batch_train_metrics(self, loss, output, label, progress, epoch):
        # Accumulate the loss over the epoch
        self.epoch_train_loss += loss
        progress.set_description("Train loss "+str(epoch)+":{:.4f}".format(loss))
        wandb.log({"Step loss": loss})

        # Calculate probabilities
        softmax = F.softmax(output, dim=1)
        #calculate max on the axis of the (different channels = number of classes)
        arg_max = torch.argmax(softmax, dim=1)
        self.epoch_train_jaccard += self.jaccard(arg_max, label)

        # Calculate accuracy
        label_flat = label.view(-1)
        output_flat = arg_max.view(-1)
        correct_pixels = torch.eq(label_flat, output_flat)
        accuracy = torch.mean(correct_pixels.float()) * 100
        self.epoch_train_accuracy += accuracy

    def log_epoch_train_metrics(self, len_train_dataloader, scheduler):
        # Calculate average per metric per epoch
        epoch_train_loss = self.epoch_train_loss / len_train_dataloader
        epoch_train_jaccard = torch.round((self.epoch_train_jaccard / len_train_dataloader) * 100, decimals=2)
        epoch_train_accuracy = torch.round(self.epoch_train_accuracy / len_train_dataloader, decimals=2)

        print(f"\n epoch train loss: {epoch_train_loss} \n")

        wandb.log({"Epoch train loss": epoch_train_loss})
        wandb.log({"Epoch train jaccard": epoch_train_jaccard})
        wandb.log({"Epoch train pixel accuracy": epoch_train_accuracy})
        wandb.log({"Learning-rate": scheduler.get_last_lr()[0]})

    def reset_epoch_validation_metrics(self):
        self.epoch_validation_jaccard = 0
        self.epoch_validation_accuracy = 0

    def log_batch_validation_metrics(self, output, label):
        # Calculate probabilities
        softmax_output = F.softmax(output, dim=1)
        # calculate max on the 3third z-axis of the tensor
        arg_max = torch.argmax(softmax_output, dim=1)
        self.epoch_validation_jaccard += self.jaccard(arg_max, label)

        # Calculate accuracy
        label_flat = label.view(-1)
        output_flat = arg_max.view(-1)
        correct_pixels = torch.eq(label_flat, output_flat)
        accuracy = torch.mean(correct_pixels.float()) * 100
        self.epoch_validation_accuracy += accuracy

    def log_epoch_validation_metrics(self, len_vali_dataloader):
        epoch_validation_jaccard = torch.round((self.epoch_validation_jaccard / len_vali_dataloader * 100), decimals=2)
        epoch_validation_accuracy = torch.round(self.epoch_validation_accuracy / len_vali_dataloader, decimals=2)
        wandb.log({"Epoch validation jaccard": epoch_validation_jaccard})
        wandb.log({"Epoch validation pixel accuracy": epoch_validation_accuracy})
