import os
from torch.utils.data import Dataset
import numpy as np


class BenGeS(Dataset):
    def __init__(
        self, data_index, esaworldcover_index, root_dir_s1, root_dir_s2, number_of_classes, bands="RGB", transform=None, normalization_value=10_000
    ):
        self.data_index = data_index
        self.esaworldcover_index = esaworldcover_index
        self.root_dir_s1 = root_dir_s1
        self.root_dir_s2 = root_dir_s2
        self.number_of_classes = number_of_classes
        self.bands = bands
        self.transform = transform
        self.normalization_value = normalization_value

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):

        #Sentinel 1
        file_name_s1 = self.data_index.loc[:, "patch_id_s1"][idx]
        path_image_s1 = os.path.join(self.root_dir_s1, file_name_s1) + "_all_bands.npy"
        img_s1 = np.load(path_image_s1)

        #Sentinel 2
        file_name_s2 = self.data_index.loc[:, "patch_id"][idx]
        path_image_s2 = os.path.join(self.root_dir_s2, file_name_s2) + "_all_bands.npy"
        img_s2 = np.load(path_image_s2)

        # Encode label
        threshold = 0.3
        label_vector = self.esaworldcover_index.loc[self.esaworldcover_index['patch_id'] == file_name_s2]
        label_vector = label_vector.drop(
            ["filename", "patch_id"], axis=1
        )
        # Set values to smaller than the threshold to 0
        label_vector = np.where(label_vector <= threshold, 0, label_vector)
        label_vector = np.squeeze(label_vector)
        # Get indexes of largest values
        max_indices = np.argpartition(label_vector, -3)[-3:]
        # Create label encoding and set to one if value is not 0
        label = np.zeros(self.number_of_classes)
        for i in range(len(max_indices)):
            if label_vector[max_indices[i]] > 0:
                label[max_indices[i]] = 1
        label = label.astype("float32")

        if self.bands == "RGB":
            img_s2 = img_s2[[3, 2, 1], :, :]
        if self.bands == "infrared":
            img_s2 = img_s2[[7, 3, 2, 1], :, :]
        if self.bands == "all":
            img_s2 = img_s2

        # change type of img
        img_s1 = img_s1.astype("float32")
        img_s2 = img_s2.astype("float32")
        img_s1_normalized = img_s1/self.normalization_value
        img_s2_normalized = img_s2/self.normalization_value

        if self.transform:
            img_s1_normalized = self.transform(img_s1_normalized)
            img_s2_normalized = self.transform(img_s2_normalized)

        # Define output tensor
        output_tensor = {
            "s1_img": img_s1_normalized,
            "s2_img": img_s2_normalized,
            "label": label
        }

        return output_tensor
