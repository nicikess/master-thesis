from datetime import datetime
import torch
import numpy as np
import wandb
from tqdm import tqdm
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
import torch.nn as nn
import torch.optim as optim


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
        self.optimizer = self.hyper_parameter.optimizer
        self.scheduler = self.hyper_parameter.scheduler
        self.loss = self.hyper_parameter.loss
        self.t_max = self.hyper_parameter.t_max
        self.eta_min = self.hyper_parameter.eta_min

        # Initialize optimier and scheduler
        self.optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)
        self.scheduler = self.scheduler(
            self.optimizer, T_max=self.t_max, eta_min=self.eta_min
        )

    def train(self):

        # Move the model to the GPU
        self.model.to(self.device)

        # Set metric
        metric_accuracy = BinaryAccuracy().to(self.device)
        metric_f1 = BinaryF1Score().to(self.device)

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

            for i, (output_tensor) in progress:
                # Transfer data to GPU if available
                images_s1 = output_tensor.get("s1_img").to(self.device)
                images_s2 = output_tensor.get("s2_img").to(self.device)
                labels = output_tensor.get("label").to(self.device)

                # Make a forward pass
                output = self.model(images_s1, images_s2)

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

                progress.set_description("Train loss epoch: {:.4f}".format(loss))

                wandb.log({"Step loss": loss})

            wandb.log({"Learning rate": self.scheduler.get_last_lr()[0]})
            self.scheduler.step()

            epoch_train_loss = epoch_train_loss / len(self.train_dl)
            epoch_train_accuracy = epoch_train_accuracy / len(self.train_dl)
            epoch_train_f1_score = epoch_train_f1_score / len(self.train_dl)

            wandb.log({"Epoch loss": epoch_train_loss})

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

                epoch_val_accuracy = epoch_val_accuracy / len(self.validation_dl)
                epoch_val_f1_score = epoch_val_f1_score / len(self.validation_dl)

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
        optimizer,
        scheduler,
        model_description,
        loss,
        t_max,
        eta_min,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_description = model_description
        self.loss = loss
        self.t_max = t_max
        self.eta_min = eta_min
