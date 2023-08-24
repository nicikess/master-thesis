import numpy as np
from rasterio.plot import show
import matplotlib.pyplot as plt
from scipy.special import softmax
from rasterio.plot import reshape_as_image
import tifffile as tiff

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

def plot_image(image_name):
    # Load the input data from the .npy file

    #Sentinel
    #image_path = '/ds2/remote_sensing/ben-ge/ben-ge/sentinel-1/'+image_name+'/'+image_name+'_all_bands'+'.npy'

    # ESA World cover
    image_path = '/ds2/remote_sensing/ben-ge/ben-ge/esaworldcover/npy/'+image_name+'_esaworldcover'+'.npy'

    #GLO
    #image_path = '/ds2/remote_sensing/ben-ge/ben-ge/glo-30_dem/npy/'+image_name+'_dem'+'.npy'

    #print(image_path)
    #tiff_image = tiff.imread(image_path)
    #input_data = np.array(tiff_image)
    input_data = np.load(image_path)

    print(input_data.shape)
    # Select and reorder the channels (assuming they are in BGR order)
    
    #Sentinel-1
    #input_data = input_data[[1], :, :]

    #Sentinel-2
    #input_data = input_data[[3,2,1], :, :]

    # Reshape the input data into an image format
    input_image = reshape_as_image(input_data)
    #print(input_image.shape)

    # Normalize the input data for display
    normalized_input = normalize_for_display(input_image)
    normalized_input = np.clip(normalized_input, 0, 1).astype(np.float32)

    # Create a plot
    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.imshow(normalized_input)
    ax.axis(False)
    plt.tight_layout()

    # Save the plot as an image file
    output_image_path = '/netscratch2/nkesseli/master-thesis-benge/src/master_thesis_benge/scripts/plot_output/sentinel-1/'
    output_image_name = '_diff_classes.png'
    plt.savefig(output_image_path+image_name+output_image_name, bbox_inches="tight", pad_inches=0.0)
    print(output_image_path+image_name+output_image_name)
    plt.show()

image_names = [

    #'S2A_MSIL2A_20170803T094031_85_66' 
    #'S1A_IW_GRDH_1SDV_20170802T163325_34TCR_85_66'
    
    'S2B_MSIL2A_20180506T105029_61_14'
    #'S1A_IW_GRDH_1SDV_20180508T055029_31UER_61_14'

    #Sentinel-2
    
    #'S2A_MSIL2A_20170818T103021_52_63',
    #'S2A_MSIL2A_20170818T103021_53_62',
    #'S2A_MSIL2A_20170818T103021_53_61',
    #'S2A_MSIL2A_20170818T103021_53_63',
    #'S2A_MSIL2A_20170818T103021_52_60',
    #'S2A_MSIL2A_20170818T103021_53_64',
    #'S2A_MSIL2A_20170818T103021_50_61',
    #'S2A_MSIL2A_20170818T103021_51_60',
    #'S2A_MSIL2A_20170818T103021_51_62',
    #'S2A_MSIL2A_20170818T103021_52_62',
    #'S2A_MSIL2A_20170818T103021_52_61',
    #'S2A_MSIL2A_20170818T103021_53_60',
    #'S2A_MSIL2A_20170818T103021_50_63',
    #'S2A_MSIL2A_20170818T103021_51_64',
    #'S2A_MSIL2A_20170818T103021_52_64',
    #'S2A_MSIL2A_20170818T103021_51_61',
    #'S2A_MSIL2A_20170818T103021_50_62',
    #'S2A_MSIL2A_20170818T103021_50_64',
    #'S2A_MSIL2A_20170818T103021_50_60',
    #'S2A_MSIL2A_20170818T103021_51_63',
    
    #Sentinel-1
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_52_63",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_53_62",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_53_61",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_53_63",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_52_60",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_53_64",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_50_61",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_51_60",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_51_62",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_52_62",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_52_61",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_53_60",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_50_63",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_51_64",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_52_64",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_51_61",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_50_62",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_50_64",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_50_60",
    #"S1A_IW_GRDH_1SDV_20170819T053446_32TMT_51_63"
]

for name in image_names:
    plot_image(name)
