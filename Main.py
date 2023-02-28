import pandas as pd
from torch.utils.data import DataLoader
from BigEarthNet import BigEarthNet
from Model import ResNetModel
from DualModel import DualResNet
from Transforms import Transforms
from Train import Train
from DisplayExampleImage import DisplayExampleImage
import numpy as np

if __name__ == '__main__':
    # Load index file
    data_index = pd.read_csv('data/test.csv_train.csv', header=None, names=["fileName"])

    # Set root dir
    root_dir = 'data/bigearthnet/bigearthnet_switzerland_data_train/'

    # Get transforms
    transforms = Transforms().transform

    # Create dataset
    train_ds = BigEarthNet(data_index, root_dir, transform=transforms, bands_rgb=True)

    # Define dataloader
    train_dl = DataLoader(train_ds, batch_size=25, shuffle=False)

    # Access data
    # for label, metainformation, img in train_dl:
        # print(img)

    # Set number of classes and load model
    #model = ResNetModel(number_of_input_channels=4, number_of_classes=19,).model
    #print(model)

    # Display example image
    # plt.imshow((out * 255).astype(np.uint8))

    # Run training routing
    # train = Train(model, train_dl)

    # Create late fusion model
    dual_model = DualResNet(in_channels_1=2, in_channels_2=4, number_of_classes=19)
    print(dual_model.res_net_1.model)
    

