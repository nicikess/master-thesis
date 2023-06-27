import numpy as np
from rasterio.plot import show
import matplotlib.pyplot as plt
from scipy.special import softmax
from rasterio.plot import reshape_as_image


def normalize_for_display(band_data):
    """Normalize multi-spectral imagery across bands.
    The input is expected to be in HxWxC format, e.g. 64x64x13.
    To account for outliers (e.g. extremly high values due to
    reflective surfaces), we normalize with the 2- and 98-percentiles
    instead of minimum and maximum of each band.
    """
    band_data = np.array(band_data)
    lower_perc = np.percentile(band_data, 2, axis=(0, 1))
    upper_perc = np.percentile(band_data, 98, axis=(0, 1))

    return (band_data - lower_perc) / (upper_perc - lower_perc)


if __name__ == "__main__":


    input = np.load('/Users/nicolaskesseli/Downloads/input.npy')
    input = input[[3,2,1],:,:]
    input = reshape_as_image(input)
    input = normalize_for_display(input)
    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.imshow(input)
    ax.axis(False)
    plt.tight_layout()
    plt.show()
    '''

    output = np.load('/Users/nicolaskesseli/Downloads/output.npy')
    # [1,11,120,120]
    softmax_output = softmax(output, axis=1)
    mask = np.argmax(softmax_output, axis=1)
    # [1,120,120]
    np.save('/Users/nicolaskesseli/Downloads/mask.npy',mask)
    
    input = reshape_as_image(mask)
    # [120,120,1]
    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.imshow(input)
    ax.axis(False)
    plt.tight_layout()
    plt.show()
    '''




