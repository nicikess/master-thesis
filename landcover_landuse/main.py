import numpy as np
from classification.display_example_image import DisplayExampleImage

if __name__ == '__main__':
    data = np.load('/Users/nicolaskesseli/Desktop/Uni/Master Thesis/example-data/s2_npy/S2A_MSIL2A_20170613T101031_3_65_all_bands.npy')

    print(np.shape(data))