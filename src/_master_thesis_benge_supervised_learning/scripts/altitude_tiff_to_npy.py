import os
import numpy as np
from PIL import Image


def tiffs_to_npy(input_folder_path, output_folder_path):
    # Get a list of all TIFF files in the input folder
    tiff_files = [
        f
        for f in os.listdir(input_folder_path)
        if f.endswith(".tiff") or f.endswith(".tif")
    ]

    for tiff_file in tiff_files:
        # Open the TIFF file using the PIL library
        tiff_image = Image.open(os.path.join(input_folder_path, tiff_file))

        # Convert the PIL image to a NumPy array
        np_array = np.array(tiff_image)

        # Save the NumPy array as a .npy file in the output folder
        np.save(os.path.join(output_folder_path, tiff_file[:-4] + ".npy"), np_array)


if __name__ == "__main__":
    input_folder = "/ds2/remote_sensing/ben-ge/ben-ge-s/glo-30_dem"
    output_folder = "/ds2/remote_sensing/ben-ge/ben-ge-s/glo-30_dem_npy"
    tiffs_to_npy(input_folder_path=input_folder, output_folder_path=output_folder)
