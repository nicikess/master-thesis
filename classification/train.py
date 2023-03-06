from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm


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
        self.model

        # Create lists for logging losses and evalualtion metrics:
        train_losses = []
        train_accs = []
        train_ious = []

        val_losses = []
        val_accs = []
        val_ious = []

        # For every epoch
        for epoch in range(50):
            epoch_loss = 0
            progress = tqdm(
                enumerate(self.train_dl), desc="Train Loss: ",
                total=len(self.train_dl)
            )

            # Specify you are in training mode
            self.model.train()

            epoch_train_loss = 0
            epoch_val_loss = 0

            epoch_train_ious = 0
            epoch_val_ious = 0

            epoch_train_accs = 0
            epoch_val_accs = 0

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

                # Accumulate the loss over the epoch
                epoch_train_loss += loss / len(self.train_dl)

                progress.set_description("Train Loss: {:.4f}".format(
                    epoch_train_loss))

            progress = tqdm(
                enumerate(self.validation_dl), desc="val Loss: ",
                total=len(self.validation_dl), position=0, leave=True, )

            # Specify you are in evaluation mode
            self.model.eval()

            # Deactivate autograd engine (no backpropagation allowed)
            with torch.no_grad():

                epoch_val_loss = 0

                for i, (labels, images) in progress:
                    # Transfer Data to GPU if available
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Make a forward pass
                    output = self.model(images)

                    # Compute the loss
                    val_loss = self.loss(output, labels)

                    # Accumulate the loss over the epoch
                    epoch_val_loss += val_loss / len(self.validation_dl)

                    progress.set_description("Validation Loss: {:.4f}".format(
                        epoch_val_loss))

            if epoch == 0:
                best_val_loss = epoch_val_loss
            else:
                if epoch_val_loss <= best_val_loss:
                    best_val_loss = epoch_val_loss
                    # Save only the best model
                    save_weights_path = "segmentation_model.pth"
                    torch.save(self.model.state_dict(), save_weights_path)

            if self.device.type == 'gpu':
                # Save losses in list, so that we can visualise them later.
                train_losses.append(epoch_train_loss.cpu().detach().numpy())
                val_losses.append(epoch_val_loss.cpu().detach().numpy())

                # Save IoUs in list, so that we can visualise them later.
                train_ious.append(epoch_train_ious.cpu().detach().numpy())
                val_ious.append(epoch_val_ious.cpu().detach().numpy())

                # Save accuracies in list, so that we can visualise them later.
                train_accs.append(epoch_train_accs.cpu().detach().numpy())
                val_accs.append(epoch_val_accs.cpu().detach().numpy())

            if self.device.type == 'cpu':
                # Save losses in list, so that we can visualise them later.
                train_losses.append(epoch_train_loss.detach().numpy())
                val_losses.append(epoch_val_loss.detach().numpy())

                # Save IoUs in list, so that we can visualise them later.
                train_ious.append(epoch_train_ious.detach().numpy())
                val_ious.append(epoch_val_ious.detach().numpy())

                # Save accuracies in list, so that we can visualise them later.
                train_accs.append(epoch_train_accs.detach().numpy())
                val_accs.append(epoch_val_accs.detach().numpy())


class HyperParameter:

    def __init__(self, epoch, batch_size, learning_rate, opt_func, milestones, weight_decay, model_description, loss):
        self.epochs = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.opt_func = opt_func
        self.milestones = milestones
        self.weight_decay = weight_decay
        self.model_description = model_description
        self.loss = loss
