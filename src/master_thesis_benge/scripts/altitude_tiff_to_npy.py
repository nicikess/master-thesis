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
        tiff_image = Image.open(
            "/Users/nicolaskesseli/Downloads/S2A_MSIL2A_20170613T101031_46_73_dem.tif"
        )

        # Convert the PIL image to a NumPy array
        np_array = np.array(tiff_image)

        # Save the NumPy array as a .npy file in the output folder
        np.save(os.path.join(output_folder_path, tiff_file[:-4] + ".npy"), np_array)


if __name__ == "__main__":
    import tifffile as tiff
    import numpy as np

    # Load TIFF image using tiff.imread()
    image = tiff.imread(
        "/Users/nicolaskesseli/Downloads/S2A_MSIL2A_20170613T101031_46_73_dem.tif"
    )

    # Convert the image to a NumPy array
    numpy_array = np.array(image)

    np.save(
        os.path.join(
            "/Users/nicolaskesseli/NICOLAS_KESSELI/Programming/Lokal/master-thesis-benge/src/master_thesis_benge/scripts",
            "S2A_MSIL2A_20170613T101031_46_73_dem" + ".npy",
        ),
        numpy_array,
    )

    # input_folder = "/Users/nicolaskesseli/Downloads"
    # output_folder = "/Users/nicolaskesseli/NICOLAS_KESSELI/Programming/Lokal/master-thesis-benge/src/master_thesis_benge/scripts"
    # tiffs_to_npy(input_folder_path=input_folder, output_folder_path=output_folder)
