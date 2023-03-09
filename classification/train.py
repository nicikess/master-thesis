from datetime import datetime
import torch
import numpy as np
import wandb
from tqdm import tqdm
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
import torch.nn as nn


class Train:
    def __init__(self, model, train_dl, validation_dl, device, wandb, hyper_parameter):
        self.model = model
        self.train_dl = train_dl
        self.validation_dl = validation_dl
        self.device = device
        self.wandb = wandb
        self.hyper_parameter = hyper_parameter
        self.epochs = self.hyper_parameter.epochs
        self.learning_rate = self.hyper_parameter.learning_rate
        self.opt_func = self.hyper_parameter.opt_func
        self.milestones = self.hyper_parameter.milestones
        self.weight_decay = self.hyper_parameter.weight_decay
        self.loss = self.hyper_parameter.loss
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

    def train(self):

        # Move the model to the GPU
        self.model.to(self.device)

        # Set metric
        metric_accuracy = BinaryAccuracy()
        metric_f1 = BinaryF1Score()

        # For every epoch
        for epoch in range(self.epochs):

            progress = tqdm(
                enumerate(self.train_dl), desc="Train Loss: ", total=len(self.train_dl)
            )

            # Specify you are in training mode
            self.model.train()

            epoch_train_loss = 0
            epoch_train_accuracy = 0
            epoch_train_f1_score = 0

            for i, (labels, images) in progress:
                # Transfer data to GPU if available
                labels = labels.to(self.device)
                images = images.to(self.device)

                # Make a forward pass
                output = self.model(images)

                # Compute the loss
                loss = self.loss(output, labels)

                # Clear the gradients
                self.optimizer.zero_grad()

                # Calculate gradients
                loss.backward()

                # Update Weights
                self.optimizer.step()

                softmax = nn.Softmax(dim=1)
                accuracy = metric_accuracy(softmax(output), labels)
                f1_score = metric_f1(softmax(output), labels)

                # Accumulate the loss over the epoch
                epoch_train_loss += loss
                epoch_train_accuracy += accuracy
                epoch_train_f1_score += f1_score

                wandb.log({"loss": loss})

            progress.set_description("Train loss epoch: {:.4f}".format(epoch_train_loss))

            epoch_train_loss = epoch_train_loss / len(self.train_dl)
            epoch_train_accuracy = epoch_train_accuracy / len(self.train_dl)
            epoch_train_f1_score = epoch_train_f1_score / len (self.train_dl)

            wandb.log({"Epoch loss": epoch_train_loss})
            wandb.log({"Epoch accuracy": epoch_train_accuracy})
            wandb.log({"Epoch f1 score": epoch_train_f1_score})

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
                epoch_val_f1_score = 0
                softmax = nn.Softmax(dim=1)

                for i, (labels, images) in progress:
                    # Transfer Data to GPU if available
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Make a forward pass
                    output = self.model(images)

                    epoch_val_accuracy += metric_accuracy(softmax(output), labels)
                    epoch_val_f1_score += metric_f1(softmax(output), labels)

                epoch_val_accuracy = epoch_train_accuracy / len(self.train_dl)
                epoch_val_f1_score = epoch_train_f1_score / len(self.train_dl)

                wandb.log({"Epoch val accuracy": epoch_val_accuracy})
                wandb.log({"Epoch val f1 score": epoch_val_f1_score})

            if epoch == 0:
                best_val = epoch_val_f1_score
            else:
                if epoch_val_f1_score <= best_val:
                    best_val = epoch_val_f1_score
                    # Save only the best model
                    save_weights_path = "classification_model.pth"
                    torch.save(self.model.state_dict(), save_weights_path)


class HyperParameter:
    def __init__(
        self,
        epochs,
        batch_size,
        learning_rate,
        opt_func,
        milestones,
        weight_decay,
        model_description,
        loss,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.opt_func = opt_func
        self.milestones = milestones
        self.weight_decay = weight_decay
        self.model_description = model_description
        self.loss = loss
