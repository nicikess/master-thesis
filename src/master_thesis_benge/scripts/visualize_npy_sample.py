import numpy as np
from rasterio.plot import show
import matplotlib.pyplot as plt
from scipy.special import softmax
from rasterio.plot import reshape_as_image

def normalize_for_display(band_data):
    """Normalize multi-spectral imagery across bands.
    The input is expected to be in HxWxC format, e.g. 64x64x13.
    To account for outliers (e.g. extremely high values due to
    reflective surfaces), we normalize with the 2- and 98-percentiles
    instead of minimum and maximum of each band.
    """
    band_data = np.array(band_data)
    lower_perc = np.percentile(band_data, 2, axis=(0, 1))
    upper_perc = np.percentile(band_data, 98, axis=(0, 1))

    return (band_data - lower_perc) / (upper_perc - lower_perc)

# Load the input data from the .npy file
input_data = np.load("/Users/nicolaskesseli/Desktop/Uni/master-thesis.nosync/data/ben-ge-s/sentinel-2/s2_npy/S2A_MSIL2A_20170613T101031_9_70_all_bands.npy")

# Select and reorder the channels (assuming they are in BGR order)
input_data = input_data[[3, 2, 1], :, :]

# Reshape the input data into an image format
input_image = reshape_as_image(input_data)

# Normalize the input data for display
normalized_input = normalize_for_display(input_image)

# Create a plot
fig, ax = plt.subplots(1, figsize=(5, 5))
ax.imshow(normalized_input)
ax.axis(False)
plt.tight_layout()
plt.show()
