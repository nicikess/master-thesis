import wandb
from torch import nn
import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)

from master_thesis_benge.supervised_baseline.training.metric import Metric

from master_thesis_benge.supervised_baseline.config.constants import (
    TRAINING_CONFIG_KEY,
    SCHEDULER_KEY,
)


class ClassificationUtils(Metric):
    # Train values
    epoch_train_loss = 0
    epoch_train_accuracy = 0
    epoch_train_accuracy_per_class = 0
    epoch_train_precision = 0
    epoch_train_recall = 0
    epoch_train_f1_score = 0
    epoch_train_f1_score_per_class = 0

    # Validation values
    epoch_val_accuracy = 0
    epoch_val_accuracy_per_class = 0
    epoch_val_precision = 0
    epoch_val_recall = 0
    epoch_val_f1_score = 0
    epoch_val_f1_per_class = 0

    def __init__(self, wandb, device, number_of_classes):
        self.wandb = wandb
        self.device = device

        # Metrics
        self.metric_accuracy_avg = BinaryAccuracy(num_classes=number_of_classes).to(
            self.device
        )
        self.metric_accuracy_per_class = BinaryAccuracy(
            multidim_average="samplewise"
        ).to(self.device)
        
        self.metric_precision_avg = BinaryPrecision(num_classes=number_of_classes).to(
            self.device
        )
        self.metric_recall_avg = BinaryRecall(num_classes=number_of_classes).to(
            self.device
        )
        self.metric_f1_avg = BinaryF1Score(num_classes=number_of_classes).to(
            self.device
        )
        self.metric_f1_avg_per_class = BinaryF1Score(multidim_average="samplewise").to(
            self.device
        )

    def calculate_loss(self, loss, output, label):
        print("output: ", output)
        print("label: ", label)
        print("output shape: ", output.shape)
        print("label shape: ", label.shape)
        print("output type: ", output.type())
        print("label type: ", label.type())
        input()
        loss = loss(output, label)
        return loss

    def reset_epoch_train_metrics(self):
        # Set metric values to 0
        self.epoch_train_loss = 0
        self.epoch_train_accuracy = 0
        self.epoch_train_accuracy_per_class = 0
        self.epoch_train_precision = 0
        self.epoch_train_recall = 0
        self.epoch_train_f1_score = 0
        self.epoch_train_f1_score_per_class = 0

    def log_batch_train_metrics(self, loss, output, label, progress, epoch):
        # Accumulate the loss over the epoch
        self.epoch_train_loss += loss
        # Calculate probabilities
        sigmoid = nn.Sigmoid()
        sigmoid_output = sigmoid(output)

        self.epoch_train_accuracy += self.metric_accuracy_avg(sigmoid_output, label)
        self.epoch_train_precision += self.metric_precision_avg(sigmoid_output, label)
        self.epoch_train_recall += self.metric_recall_avg(sigmoid_output, label)
        self.epoch_train_f1_score += self.metric_f1_avg(sigmoid_output, label)

        labels_transpose = torch.transpose(label, 0, 1)
        output_transpose = torch.transpose(sigmoid_output, 0, 1)

        self.epoch_train_accuracy_per_class += self.metric_accuracy_per_class(
            output_transpose, labels_transpose
        )
        self.epoch_train_f1_score_per_class += self.metric_f1_avg_per_class(
            output_transpose, labels_transpose
        )

        progress.set_description("Train loss "+str(epoch)+":{:.4f}".format(loss))
        wandb.log({"Step loss": loss})

    def log_epoch_train_metrics(self, len_train_dataloader, scheduler):
        print()
        print(len_train_dataloader)
        print()

        # Calculate average per metric per epoch
        epoch_train_loss = self.epoch_train_loss / len_train_dataloader
        epoch_train_accuracy = self.epoch_train_accuracy / len_train_dataloader
        epoch_train_accuracy_per_class = self.epoch_train_accuracy_per_class / len_train_dataloader
        epoch_train_f1_score_per_class = self.epoch_train_f1_score_per_class / len_train_dataloader
        self.calculate_and_log_accuracy_per_class_training(
            epoch_train_accuracy_per_class
        )
        self.calculate_and_log_f1_per_class_training(epoch_train_f1_score_per_class)
        epoch_train_precision = self.epoch_train_precision / len_train_dataloader
        epoch_train_recall = self.epoch_train_recall / len_train_dataloader
        epoch_train_f1_score = self.epoch_train_f1_score / len_train_dataloader

        print(f"\n epoch train loss: {epoch_train_loss} \n")

        wandb.log({"Epoch train loss": epoch_train_loss})
        wandb.log({"Epoch train accuracy": epoch_train_accuracy})
        wandb.log({"Epoch train precision": epoch_train_precision})
        wandb.log({"Epoch train recall": epoch_train_recall})
        wandb.log({"Epoch train f1 score": epoch_train_f1_score})
        wandb.log({"Learning-rate": scheduler.get_last_lr()[0]})

    def reset_epoch_validation_metrics(self):
        self.epoch_val_accuracy = 0
        self.epoch_val_accuracy_per_class = 0
        self.epoch_val_precision = 0
        self.epoch_val_recall = 0
        self.epoch_val_f1_score = 0
        self.epoch_val_f1_per_class = 0

    def log_batch_validation_metrics(self, output, label):
        # Calculate probabilities
        sigmoid = nn.Sigmoid()
        sigmoid_output = sigmoid(output)

        # Accuracy per class batch
        labels_transpose = torch.transpose(label, 0, 1)
        output_transpose = torch.transpose(sigmoid_output, 0, 1)

        self.epoch_val_accuracy += self.metric_accuracy_avg(sigmoid_output, label)
        self.epoch_val_precision += self.metric_precision_avg(sigmoid_output, label)
        self.epoch_val_recall += self.metric_recall_avg(sigmoid_output, label)
        self.epoch_val_f1_score += self.metric_f1_avg(sigmoid_output, label)

        self.epoch_val_accuracy_per_class += self.metric_accuracy_per_class(
            output_transpose, labels_transpose
        )
        self.epoch_val_f1_per_class += self.metric_f1_avg_per_class(
            output_transpose, labels_transpose
        )

    def log_epoch_validation_metrics(self, len_vali_dataloader):
        print()
        print(len_vali_dataloader)
        print()

        epoch_val_accuracy = self.epoch_val_accuracy / len_vali_dataloader
        epoch_val_accuracy_per_class = self.epoch_val_accuracy_per_class / len_vali_dataloader
        epoch_val_f1_per_class = self.epoch_val_f1_per_class / len_vali_dataloader
        self.calculate_and_log_accuracy_per_class_validation(
            epoch_val_accuracy_per_class
        )
        self.calculate_and_log_f1_per_class_validation(epoch_val_f1_per_class)
        epoch_val_precision = self.epoch_val_precision / len_vali_dataloader
        epoch_val_recall = self.epoch_val_recall / len_vali_dataloader
        epoch_val_f1_score = self.epoch_val_f1_score / len_vali_dataloader

        wandb.log({"Epoch val accuracy": epoch_val_accuracy})
        wandb.log({"Epoch val precision": epoch_val_precision})
        wandb.log({"Epoch val recall": epoch_val_recall})
        wandb.log({"Epoch val f1 score": epoch_val_f1_score})

    def calculate_and_log_accuracy_per_class_training(self, accuracy):
        for i in range(len(accuracy)):
            wandb.log({"Accuracy class training " + str(i): accuracy[i]})

    def calculate_and_log_accuracy_per_class_validation(self, accuracy):
        for i in range(len(accuracy)):
            wandb.log({"Accuracy class validation " + str(i): accuracy[i]})

    def calculate_and_log_f1_per_class_training(self, accuracy):
        for i in range(len(accuracy)):
            wandb.log({"F1 class training " + str(i): accuracy[i]})

    def calculate_and_log_f1_per_class_validation(self, accuracy):
        for i in range(len(accuracy)):
            wandb.log({"F1 class validation " + str(i): accuracy[i]})
