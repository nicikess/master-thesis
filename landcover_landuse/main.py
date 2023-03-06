import numpy as np
from classification.display_example_image import DisplayExampleImage
import matplotlib.pyplot as plt

if __name__ == '__main__':

    txt = input("Enter description for training run: ")

    '''
    
    data = np.load('/Users/nicolaskesseli/Desktop/Uni/Master Thesis/example-data/s2_npy/S2A_MSIL2A_20171201T112431_47_58_all_bands.npy')

    def normalize_for_display(self, band_data):
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


    def show_example_image(self):
        img_data = self.normalize_for_display(self.img)

        print(np.shape(img_data))

        # Loading the image results in a (13,64,64) shape, i.e. CxHxW. Most image libraries work with a HxWxC orientation.
        img_data = np.ma.transpose(img_data, [1, 2, 0])

        fig, ax = plt.subplots(1, figsize=(5, 5))
        ax.imshow(img_data)
        ax.axis(False)
        plt.tight_layout()

        plt.show()

    data = data[[3, 2, 1], :, :]

    data = data.astype('float32')

    DisplayExampleImage(data).show_example_image()
    
    '''
