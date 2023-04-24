import wandb
from tqdm import tqdm
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)
import torch.nn as nn
import torch

from _master_thesis_benge_supervised_learning.supervised_baseline.config.constants import (
    TRAINING_CONFIG_KEY,
    OPTIMIZER_KEY,
    LEARNING_RATE_KEY,
    SCHEDULER_KEY,
    SCHEDULER_MAX_NUMBER_ITERATIONS_KEY,
    SCHEDULER_MIN_LR_KEY,
    EPOCHS_KEY,
    LOSS_KEY,
    MODEL_CONFIG_KEY,
    MODALITIES_KEY,
    MODALITIES_LABEL_KEY,
    NUMBER_OF_CLASSES_KEY,
)


class Train:
    def __init__(self, model, train_dl, validation_dl, metrics, wandb, device, config):
        self.model = model
        self.train_dl = train_dl
        self.validation_dl = validation_dl
        self.metrics = metrics
        self.wandb = wandb
        self.device = device
        self.config = config

        # Initialize optimizer and scheduler
        self.optimizer = config[TRAINING_CONFIG_KEY][OPTIMIZER_KEY](
            model.parameters(), lr=config[TRAINING_CONFIG_KEY][LEARNING_RATE_KEY]
        )
        self.scheduler = config[TRAINING_CONFIG_KEY][SCHEDULER_KEY](
            self.optimizer,
            T_max=config[TRAINING_CONFIG_KEY][SCHEDULER_MAX_NUMBER_ITERATIONS_KEY],
            eta_min=config[TRAINING_CONFIG_KEY][SCHEDULER_MIN_LR_KEY],
        )

    def train(self):
        # Move the model to the GPU
        self.model.to(self.device)

        # Init metrics

        self.metrics = self.metrics(
            self.wandb,
            self.device,
            number_of_classes=self.config[MODEL_CONFIG_KEY][NUMBER_OF_CLASSES_KEY],
        )

        # For every epoch
        for epoch in range(self.config[TRAINING_CONFIG_KEY][EPOCHS_KEY]):
            progress = tqdm(
                enumerate(self.train_dl), desc="Train Loss: ", total=len(self.train_dl)
            )

            # Specify you are in training mode
            self.model.train()

            # Set metrics to 0
            self.metrics.reset_epoch_train_metrics()

            for i, (ben_ge_data) in progress:
                # Transfer modalities to GPU if available
                print(ben_ge_data)

                for key in ben_ge_data:
                    ben_ge_data[key] = ben_ge_data[key].to(dtype=torch.long, device=self.device)

                # Create forward data (remove label from dict)
                ben_ge_data_forward = {
                    key: ben_ge_data[key]
                    for key in self.config[TRAINING_CONFIG_KEY][MODALITIES_KEY][
                        MODALITIES_KEY
                    ]
                }

                # Rename keys
                ben_ge_data_forward = {'x{}'.format(i+1): value for i, (key, value) in enumerate(ben_ge_data_forward.items())}

                output = self.model(**ben_ge_data_forward)

                label = ben_ge_data[
                    self.config[TRAINING_CONFIG_KEY][MODALITIES_KEY][
                        MODALITIES_LABEL_KEY
                    ]
                ]

                # Compute the loss
                loss = self.config[TRAINING_CONFIG_KEY][LOSS_KEY](output, label)

                # Clear the gradients
                self.optimizer.zero_grad()

                # Calculate gradients
                loss.backward()

                # Update Weights
                self.optimizer.step()

                # Calculate batch train metrics
                self.metrics.log_batch_train_metrics(loss, output, label, progress)

            self.scheduler.step()

            # Calculate epoch train metrics
            self.metrics.log_epoch_train_metrics(len(self.train_dl), scheduler=self.scheduler)

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
                # Set validation metrics to 0
                self.metrics.reset_epoch_validation_metrics()

                for i, (ben_ge_data) in progress:
                    # Transfer modalities to GPU if available
                    for key in ben_ge_data:
                        ben_ge_data[key] = ben_ge_data[key].to(self.device)

                    # Create forward data (remove label from dict)
                    ben_ge_data_forward = {
                        key: ben_ge_data[key]
                        for key in self.config[TRAINING_CONFIG_KEY][MODALITIES_KEY][
                            MODALITIES_KEY
                        ]
                    }

                    # Rename keys
                    ben_ge_data_forward = {'x{}'.format(i + 1): value for i, (key, value) in
                                           enumerate(ben_ge_data_forward.items())}

                    # Make a forward pass
                    output = self.model(**ben_ge_data_forward)

                    label = ben_ge_data[
                        self.config[TRAINING_CONFIG_KEY][MODALITIES_KEY][
                            MODALITIES_LABEL_KEY
                        ]
                    ]

                    # Calculate batch validation metrics
                    self.metrics.log_batch_validation_metrics(output, label)

                # Calculate epoch validation metrics
                self.metrics.log_epoch_validation_metrics(len(self.validation_dl))

"""
            if self.config[OTHER_CONFIG_KEY][SAVE_MODEL_KEY]:
                if epoch == 0:
                    best_val = epoch_val_f1_score
                else:
                    if self.environment == "remote":
                        if epoch_val_f1_score <= best_val:
                            best_val = epoch_val_f1_score
                            # Save only the best model
                            run_id = str(wandb.run.id)
                            save_weights_path = (
                                "model/" + run_id + "/classification_model"
                            )
                            torch.save(self.model.state_dict(), save_weights_path)
"""
