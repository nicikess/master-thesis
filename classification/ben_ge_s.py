import os
from torch.utils.data import Dataset
import numpy as np
import json


class BenGeS(Dataset):

    def __init__(self, data_index, root_dir, number_of_classes, bands_rgb, transform=None):
        self.data_index = data_index
        self.root_dir = root_dir
        self.number_of_classes = number_of_classes
        self.bands_rgb = bands_rgb
        self.transform = transform

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_name = self.data_index.loc[:, 'patch_id'][idx]
        path_image = os.path.join(self.root_dir, file_name) + '_all_bands.npy'

        # Encode label
        label_vector = self.data_index.iloc[[idx]].drop(['filename', 'patch_id'], axis=1)
        label_vector = np.where(label_vector >= 0.5, 1, 0)
        label = np.squeeze(label_vector)
        label = label.astype('float32')

        # Read image
        img = np.load(path_image)

        if self.bands_rgb:
            img = img[[3, 2, 1], :, :]

        # change type of img
        img = img.astype('float32')

        if self.transform:
            img = self.transform(img)

        return label, img
