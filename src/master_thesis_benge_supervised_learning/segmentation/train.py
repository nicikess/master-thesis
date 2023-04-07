# We will use the tqdm library to display the progress of our training.
# from tqdm.autonotebook import tqdm
from tqdm import tqdm
from torchmetrics import JaccardIndex
import torch


class Train:
    def __init__(self, model, train_loader, val_loader, device, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

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

        # IoU
        jaccard = JaccardIndex("binary", num_classes=2)

        # For every epoch
        for epoch in range(50):
            epoch_loss = 0
            progress = tqdm(
                enumerate(self.train_loader),
                desc="Train Loss: ",
                total=len(self.train_loader),
            )

            # Specify you are in training mode
            self.model.train()

            epoch_train_loss = 0
            epoch_val_loss = 0

            epoch_train_ious = 0
            epoch_val_ious = 0

            epoch_train_accs = 0
            epoch_val_accs = 0

            for i, batch in progress:
                # Transfer data to GPU if available
                data = batch["s2_img"].float().to(self.device)
                label = batch["label"].float().to(self.device)

                # Make a forward pass
                output = self.model(data)

                # Compute the loss
                loss = self.criterion(output, label)

                # Clear the gradients
                self.optimizer.zero_grad()

                # Calculate gradients
                loss.backward()

                # Update Weights
                self.optimizer.step()

                # Accumulate the loss over the eopch
                epoch_train_loss += loss / len(self.train_loader)

                progress.set_description("Train Loss: {:.4f}".format(epoch_train_loss))

            progress = tqdm(
                enumerate(self.val_loader),
                desc="val Loss: ",
                total=len(self.val_loader),
                position=0,
                leave=True,
            )

            # Specify you are in evaluation mode
            self.model.eval()

            # Deactivate autograd engine (no backpropagation allowed)
            with torch.no_grad():
                epoch_val_loss = 0
                for j, batch in progress:
                    # Transfer Data to GPU if available
                    data = batch["s2_img"].float().to(self.device)
                    label = batch["label"].float().to(self.device)

                    # Make a forward pass
                    output = self.model(data)

                    # Compute the loss
                    val_loss = self.criterion(output, label)

                    # Accumulate the loss over the epoch
                    epoch_val_loss += val_loss / len(self.val_loader)

                    progress.set_description(
                        "Validation Loss: {:.4f}".format(epoch_val_loss)
                    )

            if epoch == 0:
                best_val_loss = epoch_val_loss
            else:
                if epoch_val_loss <= best_val_loss:
                    best_val_loss = epoch_val_loss
                    # Save only the best model
                    save_weights_path = "segmentation_model.pth"
                    torch.save(self.model.state_dict(), save_weights_path)

            # Save losses in list, so that we can visualise them later.
            train_losses.append(epoch_train_loss.cpu().detach().numpy())
            val_losses.append(epoch_val_loss.cpu().detach().numpy())

            # Save IoUs in list, so that we can visualise them later.
            train_ious.append(epoch_train_ious.cpu().detach().numpy())
            val_ious.append(epoch_val_ious.cpu().detach().numpy())

            # Save accuracies in list, so that we can visualise them later.
            train_accs.append(epoch_train_accs.cpu().detach().numpy())
            val_accs.append(epoch_val_accs.cpu().detach().numpy())
