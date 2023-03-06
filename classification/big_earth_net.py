import os
from torch.utils.data import Dataset
import numpy as np
import json


class BigEarthNet(Dataset):

    def __init__(self, data_index, root_dir, transform=None, bands_rgb=None):
        self.data_index = data_index
        self.root_dir = root_dir
        self.transform = transform
        self.bandsRGB = bands_rgb

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_name = self.data_index.loc[:, 'fileName'][idx]
        path_folder = os.path.join(self.root_dir, file_name) + '_all_bands.npy'
        path_image = os.path.join(path_folder, file_name) + '_all_bands.npy'
        path_label = os.path.join(path_folder, file_name) + '_labels.npy'
        path_meta_information = os.path.join(path_folder, file_name) + '_labels_metadata.json'

        # read label
        label = np.load(path_label)

        # read image
        img = np.load(path_image)

        if self.bandsRGB == True:
            img = img[[3, 2, 1], :, :]

        # read meta information
        with open(path_meta_information, 'r') as j:
            meta_information = json.loads(j.read())

        # change type of img
        img = img.astype('float32')

        if self.transform:
            img = self.transform(img)

        return label, meta_information, img
