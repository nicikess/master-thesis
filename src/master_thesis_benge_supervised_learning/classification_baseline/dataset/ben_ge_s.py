import os

from torch.utils.data import Dataset
import numpy as np

from master_thesis_benge_supervised_learning.classification_baseline.config.config import *
from master_thesis_benge_supervised_learning.classification_baseline.config.constants import *


class BenGeS(Dataset):
    def __init__(
        self,
        esa_world_cover_data,
        sentinel_1_2_metadata,
        era5_data,
        root_dir_s1,
        root_dir_s2,
        root_dir_world_cover,
        wandb,
    ):
        self.data_index = esa_world_cover_data
        self.sentinel_1_2_metadata = sentinel_1_2_metadata
        self.era5_data = era5_data
        self.root_dir_s1 = root_dir_s1
        self.root_dir_s2 = root_dir_s2
        self.root_dir_world_cover = root_dir_world_cover

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):

        # Sentinel 2
        file_name_s2 = self.data_index.loc[:, "patch_id"][idx]
        path_image_s2 = os.path.join(self.root_dir_s2, file_name_s2) + "_all_bands.npy"
        img_s2 = np.load(path_image_s2)

        # Load other modalities

        # Sentinel 1
        #file_name_s1 = self.sentinel_1_2_metadata.loc[self.sentinel_1_2_metadata['patch_id'] == file_name_s2, "patch_id_s1"].values[0]
        #path_image_s1 = os.path.join(self.root_dir_s1, file_name_s1) + "_all_bands.npy"
        #img_s1 = np.load(path_image_s1)

        # World cover
        #file_name_world_cover = self.data_index.loc[:, "patch_id"][idx]
        #path_image_world_cover = (os.path.join(self.root_dir_world_cover, file_name_world_cover) + "_esaworldcover.npy")
        #img_world_cover = np.load(path_image_world_cover)

        # Encode label
        threshold = config.get(label_threshold_key)
        label_vector = self.data_index.loc[[idx]]
        label_vector = label_vector.drop(["filename", "patch_id"], axis=1)
        # Set values to smaller than the threshold to 0
        label_vector = np.where(label_vector <= threshold, 0, label_vector)
        label_vector = np.squeeze(label_vector)
        # Get indexes of largest values
        max_indices = np.argpartition(label_vector, -3)[-3:]
        # Create label encoding and set to one if value is not 0
        label = np.zeros(config.get(number_of_classes_key))
        for i in range(len(max_indices)):
            if label_vector[max_indices[i]] > 0:
                label[max_indices[i]] = 1
        label = label.astype(NUMPY_DTYPE)

        if config.get(bands_key) == Bands.RGB:
            img_s2 = img_s2[[3, 2, 1], :, :]
        if config.get(bands_key) == Bands.INFRARED:
            img_s2 = img_s2[[7, 3, 2, 1], :, :]
        if config.get(bands_key) == Bands.ALL:
            img_s2 = img_s2

        # change type of img
        #img_s1 = img_s1.astype(NUMPY_DTYPE)
        img_s2 = img_s2.astype(NUMPY_DTYPE)

        #Clip values to 0-10000
        img_s2 = np.clip(img_s2, 0, 10000)

        #img_world_cover = img_world_cover.astype(NUMPY_DTYPE)
        #img_s1_normalized = img_s1 / self.normalization_value
        img_s2_normalized = img_s2 / config.get(normalization_value_key)
        #img_world_cover_normalized = img_world_cover  # /self.normalization_value

        if config.get(transforms_key):
            #img_s1_normalized = self.transform(img_s1_normalized)
            img_s2_normalized = config.get(transforms_key)(img_s2_normalized)
            #img_world_cover_normalized = self.transform(img_world_cover_normalized)

        # Define output tensor
        ben_ge_data = {
            #"s1_img": img_s1_normalized,
            "s2_img": img_s2_normalized,
            #"world_cover_img": img_world_cover_normalized,
            "label": label,
        }

        return ben_ge_data
