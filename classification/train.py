from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm


class Train:

    def __init__(self, model, train_dl):

        self.hyper_parameter = HyperParameter()
        self.model = model
        self.train_dl = train_dl
        self.epochs = self.hyper_parameter.epochs
        self.learning_rate = self.hyper_parameter.learning_rate
        self.opt_func = self.hyper_parameter.opt_func
        self.milestones = self.hyper_parameter.milestones
        self.weight_decay = self.hyper_parameter.weight_decay
        self.loss = self.hyper_parameter.loss

        # init collection of training epoch losses
        train_epoch_losses = []

        # set the model in training mode
        self.model.train()

        self.optimizer = self.opt_func(self.model.fc.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)

        # train the CIFAR10 model
        for epoch in tqdm(range(self.epochs)):

            # init collection of mini-batch losses
            train_mini_batch_losses = []

            # iterate over all-mini batches
            for i, (labels, meta_information, images) in tqdm(enumerate(train_dl)):
                # push mini-batch data to computation device
                # images = images.to(device)
                # labels = labels.to(device)

                # run forward pass through the network
                output = self.model(images)

                # reset graph gradients
                self.optimizer.zero_grad()

                # determine classification loss
                loss = self.loss(output, labels)

                # wandb.log({"loss": loss})

                # run backward pass
                loss.backward()

                # update network paramaters
                self.optimizer.step()

                # collect mini-batch reconstruction loss
                train_mini_batch_losses.append(loss.data.item())

                # print(f'loss {loss}')

            # determine mean min-batch loss of epoch
            train_epoch_loss = np.mean(train_mini_batch_losses)

            # print('Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            self.scheduler.step()

            # print('Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            # Added
            # result = evaluate(model, vali_dataloader)

            # print epoch loss
            now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
            print('[LOG {}] epoch: {} train-loss: {}'.format(str(now), str(epoch), str(train_epoch_loss)))

            if (epoch % 10 == 0):
                # set filename of actual model
                model_name = 'challenge_model_epoch_{}.pth'.format(str(epoch))
                # save current model to GDrive models directory
                print(f'Here it would save {model_name}')
                # torch.save(model.state_dict(), os.path.join(MODEL_PATH, model_name))

            # determine mean min-batch loss of epoch
            train_epoch_losses.append(train_epoch_loss)


class HyperParameter:

    def __init__(self):

        self.epochs = 20
        self.batch_size = 32
        self.learning_rate = 0.001
        self.opt_func = torch.optim.Adam
        self.milestones = [5, 15]
        self.weight_decay = 0
        self.description = "resnet50_no_transformation_with_normalisation"
        self.loss = torch.nn.CrossEntropyLoss()
