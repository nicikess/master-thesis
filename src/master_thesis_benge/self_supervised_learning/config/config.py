class Hparams:
    def __init__(self):
        self.epochs = 200  #
        self.seed = 77777
        self.cuda = True
        self.img_size = 120  # img shape
        self.save = "./saved_models/"  # save checkpoint
        self.load = False  # load pretrained checkpoint
        self.gradient_accumulation_steps = 5
        # self.batch_size = 200
        self.lr = 3e-4  # for Adam only
        self.weight_decay = 1e-6
        self.embedding_size = 128
        # self.temperature = 0.5
        self.checkpoint_path = "./SimCLR_ResNet18.ckpt"  # replace checkpoint path here
