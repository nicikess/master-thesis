import pytorch_lightning as pl
import wandb
from torch.optim import Adam

# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from master_thesis_benge.self_supervised_learning.loss.contrastive_loss import ContrastiveLoss

from master_thesis_benge.self_supervised_learning.config.constants import (
    TRAINING_CONFIG_KEY,
    EMEDDING_SIZE_KEY,
    WEIGHT_DECAY_KEY,
    GRADIENT_ACCUMULATION_STEPS_KEY,
    LEARNING_RATE_KEY,
    PROJECTION_HEAD_KEY,
    FEATURE_DIMENSION_KEY
)

def define_param_groups(model, weight_decay, optimizer_name):
    def exclude_from_wd_and_adaptation(name):
        if "bn" in name:
            return True
        if optimizer_name == "lars" and "bias" in name:
            return True

    param_groups = [
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if not exclude_from_wd_and_adaptation(name)
            ],
            "weight_decay": weight_decay,
            "layer_adaptation": True,
        },
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if exclude_from_wd_and_adaptation(name)
            ],
            "weight_decay": 0.0,
            "layer_adaptation": False,
        },
    ]
    return param_groups

class SimCLR_pl(pl.LightningModule):
    def __init__(self, training_config, feat_dim=512, in_channels_1=None, in_channels_2=None, in_channels_3=None):
        super().__init__()
        self.training_config = training_config
        self.model_modality_1 = self.training_config[TRAINING_CONFIG_KEY][PROJECTION_HEAD_KEY](
                in_channels=in_channels_1,
                embedding_size = self.training_config[TRAINING_CONFIG_KEY][EMEDDING_SIZE_KEY], 
                mlp_dim=training_config[TRAINING_CONFIG_KEY][FEATURE_DIMENSION_KEY]
            )
        self.model_modality_2 = self.training_config[TRAINING_CONFIG_KEY][PROJECTION_HEAD_KEY](
            in_channels=in_channels_2,
            embedding_size = self.training_config[TRAINING_CONFIG_KEY][EMEDDING_SIZE_KEY],
            mlp_dim=training_config[TRAINING_CONFIG_KEY][FEATURE_DIMENSION_KEY]
            )
        self.model_modality_3 = self.training_config[TRAINING_CONFIG_KEY][PROJECTION_HEAD_KEY](
            in_channels=in_channels_3,
            embedding_size = self.training_config[TRAINING_CONFIG_KEY][EMEDDING_SIZE_KEY],
            mlp_dim=training_config[TRAINING_CONFIG_KEY][FEATURE_DIMENSION_KEY]
            )
        self.loss = ContrastiveLoss(
            wandb.config.batch_size, temperature=wandb.config.temperature
        )
        wandb.log({"batch size": wandb.config.batch_size})
        wandb.log({"temperature": wandb.config.temperature})

        print("disabling automatic optimization")
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        modality_1 = batch[int(wandb.config.modalities[0])]
        modality_2 = batch[int(wandb.config.modalities[1])]
        modality_3 = batch[int(wandb.config.modalities[2])]
        
        z1 = self.model_modality_1(modality_1)
        z2 = self.model_modality_2(modality_2)
        z3 = self.model_modality_3(modality_3)

        loss_z1_z2 = self.loss(z1, z2)
        loss_z1_z3 = self.loss(z1, z3)
        loss_z2_z3 = self.loss(z2, z3)

        loss = (loss_z1_z2 + loss_z1_z3 + loss_z2_z3) / 3

        self.log("Contrastive loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"loss batch": loss})

        opt_modality_1, opt_modality_2, opt_modality_3 = self.optimizers()  # Access optimizers

        self.manual_backward(loss)  # Perform manual backward pass

        if (batch_idx + 1) % self.training_config[TRAINING_CONFIG_KEY][GRADIENT_ACCUMULATION_STEPS_KEY] == 0:
            opt_modality_1.step()
            opt_modality_2.step()
            opt_modality_3.step()
            opt_modality_1.zero_grad()
            opt_modality_2.zero_grad()
            opt_modality_3.zero_grad()

        return loss

    def configure_optimizers(self):
        param_groups_modality_1 = define_param_groups(self.model_modality_1, self.training_config[TRAINING_CONFIG_KEY][WEIGHT_DECAY_KEY], "adam")
        param_groups_modality_2 = define_param_groups(self.model_modality_2, self.training_config[TRAINING_CONFIG_KEY][WEIGHT_DECAY_KEY], "adam")
        param_groups_modality_3 = define_param_groups(self.model_modality_3, self.training_config[TRAINING_CONFIG_KEY][WEIGHT_DECAY_KEY], "adam")
        lr = self.training_config[TRAINING_CONFIG_KEY][LEARNING_RATE_KEY]

        optimizer_s1 = Adam(param_groups_modality_1, lr=lr, weight_decay=self.training_config[TRAINING_CONFIG_KEY][WEIGHT_DECAY_KEY])
        optimizer_s2 = Adam(param_groups_modality_2, lr=lr, weight_decay=self.training_config[TRAINING_CONFIG_KEY][WEIGHT_DECAY_KEY])
        optimizer_s3 = Adam(param_groups_modality_3, lr=lr, weight_decay=self.training_config[TRAINING_CONFIG_KEY][WEIGHT_DECAY_KEY])


        print(
            f"Optimizer Adam, "
            f"Learning Rate {lr}, "
            f"Effective batch size {wandb.config.batch_size * self.training_config[TRAINING_CONFIG_KEY][GRADIENT_ACCUMULATION_STEPS_KEY]}"
        )

        return [optimizer_s1, optimizer_s2, optimizer_s3]