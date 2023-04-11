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

from master_thesis_benge_supervised_learning.classification_baseline.config.constants import *
from master_thesis_benge_supervised_learning.classification_baseline.training.train_utils import TrainUtils

class Train:
    def __init__(
        self,
        model,
        train_dl,
        validation_dl,
        wandb,
        device,
        config
    ):
        self.model = model
        self.train_dl = train_dl
        self.validation_dl = validation_dl
        self.wandb = wandb,
        self.device = device
        self.config = config

        # Initialize optimizer and scheduler
        self.optimizer = config[TRAINING_CONFIG_KEY][OPTIMIZER_KEY](model.parameters(), lr=config[TRAINING_CONFIG_KEY][LEARNING_RATE_KEY])
        self.scheduler = config[TRAINING_CONFIG_KEY][SCHEDULER_KEY](self.optimizer, T_max=config[TRAINING_CONFIG_KEY][SCHEDULER_MAX_NUMBER_ITERATIONS_KEY], eta_min=config[TRAINING_CONFIG_KEY][SCHEDULER_MIN_LR_KEY])

    def train(self):

        # Move the model to the GPU
        self.model.to(self.device)

        # Set metrics
        metric_accuracy_avg = BinaryAccuracy(num_classes=self.config[TRAINING_CONFIG_KEY][SCHEDULER_KEY]).to(self.device)
        metric_accuracy_per_class = BinaryAccuracy(multidim_average='samplewise').to(self.device)
        metric_precision_avg = BinaryPrecision(num_classes=self.config[TRAINING_CONFIG_KEY][SCHEDULER_KEY]).to(self.device)
        metric_recall_avg = BinaryRecall(num_classes=self.config[TRAINING_CONFIG_KEY][SCHEDULER_KEY]).to(self.device)
        metric_f1_avg = BinaryF1Score(num_classes=self.config[TRAINING_CONFIG_KEY][SCHEDULER_KEY]).to(self.device)
        metric_f1_avg_per_class = BinaryF1Score(multidim_average='samplewise').to(self.device)


        # For every epoch
        for epoch in range(self.config[TRAINING_CONFIG_KEY][EPOCHS_KEY]):

            progress = tqdm(
                enumerate(self.train_dl), desc="Train Loss: ", total=len(self.train_dl)
            )

            # Specify you are in training mode
            self.model.train()

            epoch_train_loss = 0
            epoch_train_accuracy = 0
            epoch_train_accuracy_per_class = 0
            epoch_train_precision = 0
            epoch_train_recall = 0
            epoch_train_f1_score = 0
            epoch_train_f1_score_per_class = 0

            for i, (ben_ge_data) in progress:

                # Transfer data to GPU if available
                s2_images = ben_ge_data[S2_IMG_KEY].to(self.device)
                labels = ben_ge_data[MULTICLASS_LABEL_KEY].to(self.device)
                if self.config[MODEL_CONFIG_KEY][MULTI_MODAL_KEY]:
                    # Transfer other data modalities to GPU if available
                    s1_images = ben_ge_data[S1_IMG_KEY].to(self.device)
                    #world_cover_images = ben_ge_data[WORLD_COVER_IMG_KEY].to(self.device)
                    altitude_images = ben_ge_data[ALTITUDE_IMG_KEY].to(self.device)


                # Make a forward pass
                if self.config[MODEL_CONFIG_KEY][MULTI_MODAL_KEY]:
                    output = self.model(s1_images, s2_images, altitude_images)
                else:
                    output = self.model(s2_images)

                # Compute the loss
                loss = self.config[TRAINING_CONFIG_KEY][LOSS_KEY](output, labels)

                # Clear the gradients
                self.optimizer.zero_grad()

                # Calculate gradients
                loss.backward()

                # Update Weights
                self.optimizer.step()

                # Calculate probabilities
                sigmoid = nn.Sigmoid()
                sigmoid_output = sigmoid(output)

                # Accumulate the loss over the epoch
                epoch_train_loss += loss

                # Accuracy per class batch
                labels_transpose = torch.transpose(labels, 0, 1)
                output_transpose = torch.transpose(sigmoid_output, 0, 1)

                epoch_train_accuracy += metric_accuracy_avg(sigmoid_output, labels)
                epoch_train_precision += metric_precision_avg(sigmoid_output, labels)
                epoch_train_recall += metric_recall_avg(sigmoid_output, labels)
                epoch_train_f1_score += metric_f1_avg(sigmoid_output, labels)

                epoch_train_accuracy_per_class += metric_accuracy_per_class(output_transpose, labels_transpose)
                epoch_train_f1_score_per_class += metric_f1_avg_per_class(output_transpose, labels_transpose)

                progress.set_description("Train loss epoch: {:.4f}".format(loss))
                wandb.log({"Step loss": loss})

            #wandb.log({"Learning-rate": self.scheduler.get_last_lr()[0]})
            self.scheduler.step()

            # Calculate average per metric per epoch
            epoch_train_loss = epoch_train_loss / len(self.train_dl)
            epoch_train_accuracy = epoch_train_accuracy / len(self.train_dl)
            epoch_train_accuracy_per_class = epoch_train_accuracy_per_class / len(self.train_dl)
            epoch_train_f1_score_per_class = epoch_train_f1_score_per_class / len(self.train_dl)
            TrainUtils.caluculate_and_log_accuracy_per_class_training(epoch_train_accuracy_per_class)
            TrainUtils.caluculate_and_log_f1_per_class_training(epoch_train_f1_score_per_class)
            epoch_train_precision = epoch_train_precision / len(self.train_dl)
            epoch_train_recall = epoch_train_recall / len(self.train_dl)
            epoch_train_f1_score = epoch_train_f1_score / len(self.train_dl)

            print(f'\n epoch train loss: {epoch_train_loss} \n')

            wandb.log({"Epoch train loss": epoch_train_loss})
            wandb.log({"Epoch train accuracy": epoch_train_accuracy})
            wandb.log({"Epoch train precision": epoch_train_precision})
            wandb.log({"Epoch train recall": epoch_train_recall})
            wandb.log({"Epoch train f1 score": epoch_train_f1_score})

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
                epoch_val_precision = 0
                epoch_val_recall = 0
                epoch_val_f1_score = 0
                epoch_val_f1_per_class = 0

                for i, (ben_ge_data) in progress:

                    # Transfer data to GPU if available
                    s2_images = ben_ge_data[S2_IMG_KEY].to(self.device)
                    labels = ben_ge_data[MULTICLASS_LABEL_KEY].to(self.device)
                    if self.config[MODEL_CONFIG_KEY][MULTI_MODAL_KEY]:
                        s1_images = ben_ge_data[S1_IMG_KEY].to(self.device)
                        #world_cover_images = ben_ge_data[WORLD_COVER_IMG_KEY].to(self.device)
                        altitude_images = ben_ge_data[ALTITUDE_IMG_KEY].to(self.device)

                    # Make a forward pass
                    if self.config[MODEL_CONFIG_KEY][MULTI_MODAL_KEY]:
                        output = self.model(s1_images, s2_images, altitude_images)
                    else:
                        output = self.model(s2_images)

                    # Calculate probabilities
                    sigmoid = nn.Sigmoid()
                    sigmoid_output = sigmoid(output)

                    # Accuracy per class batch
                    labels_transpose = torch.transpose(labels, 0, 1)
                    output_transpose = torch.transpose(sigmoid_output, 0, 1)

                    epoch_val_accuracy += metric_accuracy_avg(sigmoid_output, labels)
                    epoch_val_precision += metric_precision_avg(sigmoid_output, labels)
                    epoch_val_recall += metric_recall_avg(sigmoid_output, labels)
                    epoch_val_f1_score += metric_f1_avg(sigmoid_output, labels)

                    epoch_val_accuracy_per_class += metric_accuracy_per_class(output_transpose, labels_transpose)
                    epoch_val_f1_per_class += metric_f1_avg_per_class(output_transpose, labels_transpose)

                epoch_val_accuracy = epoch_val_accuracy / len(self.validation_dl)
                epoch_val_accuracy_per_class = epoch_val_accuracy_per_class / len(self.validation_dl)
                epoch_val_f1_per_class = epoch_val_f1_per_class / len(self.validation_dl)
                TrainUtils.caluculate_and_log_accuracy_per_class_validation(epoch_val_accuracy_per_class)
                TrainUtils.caluculate_and_log_f1_per_class_validation(epoch_val_f1_per_class)
                epoch_val_precision = epoch_val_precision / len(self.validation_dl)
                epoch_val_recall = epoch_val_recall / len(self.validation_dl)
                epoch_val_f1_score = epoch_val_f1_score / len(self.validation_dl)

                wandb.log({"Epoch val accuracy": epoch_val_accuracy})
                wandb.log({"Epoch val precision": epoch_val_precision})
                wandb.log({"Epoch val recall": epoch_val_recall})
                wandb.log({"Epoch val f1 score": epoch_val_f1_score})


            if self.config[OTHER_CONFIG_KEY][SAVE_MODEL_KEY]:
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
