class SimCLR_pl(pl.LightningModule):
    def __init__(self, config, model=None, feat_dim=512):
        super().__init__()
        self.config = config
        self.model = AddProjection(config, model=model, mlp_dim=feat_dim)
        self.loss = ContrastiveLoss(wandb.config.batch_size, temperature=wandb.config.temperature)
        wandb.log({"batch size": wandb.config.batch_size})
        wandb.log({"temperature": wandb.config.temperature})

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        (x1, x2), labels = batch
        z1 = self.model(x1)
        z2 = self.model(x2)
        loss = self.loss(z1, z2)
        self.log('Contrastive loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"loss batch": loss})
        return loss

    def define_param_groups(model, weight_decay, optimizer_name):
        def exclude_from_wd_and_adaptation(name):
            if 'bn' in name:
                return True
            if optimizer_name == 'lars' and 'bias' in name:
                return True

        param_groups = [
            {
                'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
                'weight_decay': weight_decay,
                'layer_adaptation': True,
            },
            {
                'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
                'weight_decay': 0.,
                'layer_adaptation': False,
            },
        ]
        return param_groups

    def configure_optimizers(self):
        max_epochs = int(self.config.epochs)
        param_groups = self.define_param_groups(self.model, self.config.weight_decay, 'adam')
        lr = self.config.lr
        optimizer = Adam(param_groups, lr=lr, weight_decay=self.config.weight_decay)

        print(f'Optimizer Adam, '
              f'Learning Rate {lr}'
              f'Effective batch size {wandb.config.batch_size * self.config.gradient_accumulation_steps}')

        scheduler_warmup = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=max_epochs,
                                                         warmup_start_lr=0.0)

        return [optimizer], [scheduler_warmup]