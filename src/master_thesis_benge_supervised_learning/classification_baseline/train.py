from datetime import datetime
import torch
import numpy as np
import wandb
from tqdm import tqdm
from torchmetrics.classification import (
    BinaryAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
import torch.nn as nn

from src.master_thesis_benge_supervised_learning.constants import TrainingParameters

class Train:
    def __init__(
        self,
        model,
        train_dl,
        validation_dl,
        number_of_classes,
        device,
        wandb,
        hyper_parameter,
        environment,
        multi_modal
    ):
        self.model = model
        self.train_dl = train_dl
        self.validation_dl = validation_dl
        self.number_of_classes = number_of_classes
        self.device = device
        self.wandb = wandb
        self.hyper_parameter = hyper_parameter
        self.environment = environment
        self.multi_modal = multi_modal
        self.epochs = self.hyper_parameter.epochs
        self.learning_rate = self.hyper_parameter.learning_rate
        self.optimizer = self.hyper_parameter.optimizer
        self.scheduler = self.hyper_parameter.scheduler
        self.loss = self.hyper_parameter.loss

        # Initialize optimizer and scheduler
        self.optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)
        self.scheduler = self.scheduler(self.optimizer, T_max=TrainingParameters.SCHEDULER_MAX_NUMBER_INTERATIONS.value, eta_min=TrainingParameters.SCHEDULER_MIN_LR.value)

    def train(self):

        # Move the model to the GPU
        self.model.to(self.device)

        # Set metrics
        # metric_accuracy_avg = MulticlassAccuracy(num_classes=self.number_of_classes).to(self.device)
        metric_accuracy = BinaryAccuracy().to(self.device)
        metric_accuracy_per_class = BinaryAccuracy(multidim_average='samplewise').to(self.device)
        # metric_precision_avg = MulticlassPrecision(num_classes=self.number_of_classes).to(self.device)
        # metric_precision = MulticlassPrecision(num_classes=self.number_of_classes, average=None).to(self.device)
        # metric_recall_avg = MulticlassRecall(num_classes=self.number_of_classes).to(self.device)
        # metric_recall = MulticlassRecall(num_classes=self.number_of_classes, average=None).to(self.device)
        # metric_f1_avg = MulticlassRecall(num_classes=self.number_of_classes).to(self.device)
        # metric_f1 = MulticlassRecall(num_classes=self.number_of_classes, average=None).to(self.device)

        # For every epoch
        for epoch in range(self.epochs):

            progress = tqdm(
                enumerate(self.train_dl), desc="Train Loss: ", total=len(self.train_dl)
            )

            # Specify you are in training mode
            self.model.train()

            epoch_train_loss = 0
            epoch_train_accuracy = 0
            epoch_train_accuracy_per_class = 0
            #epoch_train_precision = 0
            #epoch_train_recall = 0
            #epoch_train_f1_score = 0

            for i, (ben_ge_data) in progress:

                # Transfer data to GPU if available
                s2_images = ben_ge_data["s2_img"].to(self.device)
                labels = ben_ge_data["label"].to(self.device)
                if self.multi_modal:
                    # Transfer other data modalities to GPU if available
                    s1_images = ben_ge_data["s1_img"].to(self.device)

                # Make a forward pass
                if self.multi_modal:
                    output = self.model(s1_images, s2_images)
                else:
                    output = self.model(s2_images)

                # Compute the loss
                loss = self.loss(output, labels)
                print(loss)

                # Clear the gradients
                self.optimizer.zero_grad()

                # Calculate gradients
                loss.backward()

                # Update Weights
                self.optimizer.step()

                # Calculate probabilities
                softmax = nn.Softmax(dim=1)
                softmax_output = softmax(output)

                # Accumulate the loss over the epoch
                epoch_train_loss += loss

                # Overall accuracy batch
                epoch_train_accuracy += metric_accuracy(softmax_output, labels)

                # Accuracy per class batch
                labels_transpose = torch.transpose(labels, 0, 1)
                output_transpose = torch.transpose(softmax_output, 0, 1)
                epoch_train_accuracy_per_class += metric_accuracy_per_class(output_transpose, labels_transpose)
                # epoch_train_precision += metric_precision_avg(softmax_output, labels)
                # epoch_train_recall += metric_recall_avg(softmax_output, labels)
                # epoch_train_f1_score += metric_f1_avg(softmax_output, labels)

                progress.set_description("Train loss epoch: {:.4f}".format(loss))
                #wandb.log({"Step loss": loss})

            # TODO - check if scheduler works correct
            wandb.log({"Learning-rate": self.scheduler.get_last_lr()[0]})
            self.scheduler.step()

            # Calculate average per metric per epoch
            epoch_train_loss = epoch_train_loss / len(self.train_dl)
            epoch_train_accuracy = epoch_train_accuracy / len(self.train_dl)
            epoch_train_accuracy_per_class = epoch_train_accuracy_per_class / len(self.train_dl)
            # epoch_train_precision = epoch_train_precision / len(self.train_dl)
            # epoch_train_recall = epoch_train_recall / len(self.train_dl)
            # epoch_train_f1_score = epoch_train_f1_score / len(self.train_dl)

            wandb.log({"Epoch train loss": epoch_train_loss})
            wandb.log({"Epoch train accuracy": epoch_train_accuracy})
            #wandb.log({"Epoch train accuracy per class": epoch_train_accuracy_per_class})

            #wandb.log({"Epoch train precision accuracy": epoch_train_precision})
            #wandb.log({"Epoch train recall accuracy": epoch_train_recall})
            #wandb.log({"Epoch train f1 score": epoch_train_f1_score})

            progress = tqdm(
                enumerate(self.validation_dl),
                desc="val Loss: ",
                total=len(self.validation_dl),
                position=0,
                leave=True,
            )

            # Specify you are in evaluation mode
            self.model.eval()

            # Deactivate autograd engine (no backpropagation allowed)
            with torch.no_grad():

                epoch_val_accuracy = 0
                epoch_val_accuracy_per_class = 0
                #epoch_val_precision = 0
                #epoch_val_recall = 0
                #epoch_val_f1_score = 0

                for i, (ben_ge_data) in progress:

                    # Transfer data to GPU if available
                    s2_images = ben_ge_data["s2_img"].to(self.device)
                    labels = ben_ge_data["label"].to(self.device)
                    if self.multi_modal:
                        s1_images = ben_ge_data["s1_img"].to(self.device)

                    # Make a forward pass
                    output = self.model(s2_images)
                    if self.multi_modal:
                        output = self.model(s1_images, s2_images)

                    # Calculate probabilities
                    softmax = nn.Softmax(dim=1)
                    softmax_output = softmax(output)

                    epoch_val_accuracy += metric_accuracy(softmax_output, labels)
                    epoch_val_accuracy_per_class += metric_accuracy_per_class(softmax_output, labels)
                    #epoch_val_precision += metric_precision_avg(softmax_output, labels)
                    #epoch_val_recall += metric_recall_avg(softmax_output, labels)
                    #epoch_val_f1_score += metric_f1_avg(softmax_output, labels)

                epoch_val_accuracy = epoch_val_accuracy / len(self.validation_dl)
                epoch_val_accuracy_per_class = epoch_val_accuracy_per_class / len(self.validation_dl)
                #epoch_val_precision = epoch_val_precision / len(self.val_dl)
                #epoch_val_recall = epoch_val_recall / len(self.val_dl)
                #epoch_val_f1_score = epoch_val_f1_score / len(self.val_dl)

                wandb.log({"Epoch val accuracy": epoch_val_accuracy})
                wandb.log({"Epoch val accuracy per class": epoch_val_accuracy_per_class})

                #wandb.log({"Epoch val precision accuracy": epoch_val_precision})
                #wandb.log({"Epoch val recall accuracy": epoch_val_recall})
                #wandb.log({"Epoch val f1 score": epoch_val_f1_score})


            # TODO - uncomment, once f1 score is implemented

            '''
            if epoch == 0:
                best_val = epoch_val_f1_score
            else:
                if self.environment == "remote":
                    if epoch_val_f1_score <= best_val:
                        best_val = epoch_val_f1_score
                        # Save only the best model
                        run_id = str(wandb.run.id)
                        save_weights_path = "model/" + run_id + "/classification_model"
                        torch.save(self.model.state_dict(), save_weights_path)
            '''

class HyperParameter:
    def __init__(
        self,
        epochs,
        batch_size,
        learning_rate,
        optimizer,
        scheduler,
        loss,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
