def train():
    run = wandb.init()
    available_gpus = len(
        [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    )
    save_model_path = os.path.join(os.getcwd(), "saved_models/")
    print("available_gpus:", available_gpus)
    filename = "SimCLR_ResNet18_adam_"
    resume_from_checkpoint = False
    train_config = Hparams()

    reproducibility(train_config)
    save_name = filename + ".ckpt"

    model = SimCLR_pl(train_config, model=resnet18(pretrained=False), feat_dim=512)

    transform = Augment(train_config.img_size)
    data_loader = get_str1_dataloader_Euro(wandb.config.batch_size, transform)

    accumulator = GradientAccumulationScheduler(
        scheduling={0: train_config.gradient_accumulation_steps}
    )
    checkpoint_callback = ModelCheckpoint(
        filename=filename,
        dirpath=save_model_path,
        save_last=True,
        save_top_k=2,
        monitor="Contrastive loss_epoch",
        mode="min",
    )

    if resume_from_checkpoint:
        trainer = Trainer(
            callbacks=[accumulator, checkpoint_callback],
            gpus=available_gpus,
            max_epochs=train_config.epochs,
            resume_from_checkpoint=train_config.checkpoint_path,
        )
    else:
        trainer = Trainer(
            callbacks=[accumulator, checkpoint_callback],
            gpus=available_gpus,
            max_epochs=train_config.epochs,
        )

    trainer.fit(model, data_loader)

    trainer.save_checkpoint(save_name)
    from google.colab import files

    files.download(save_name)
