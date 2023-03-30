class DistributionDataset:
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
    ):
        self.model = model
        self.train_dl = train_dl
        self.validation_dl = validation_dl
