from remote_sensing_core.transforms.ffcv.min_max_scaler import MinMaxScaler
from remote_sensing_core.transforms.ffcv.clipping import Clipping
from remote_sensing_core.transforms.ffcv.channel_selector import ChannelSelector
from remote_sensing_core.transforms.ffcv.convert import Convert
from remote_sensing_core.transforms.ffcv.esa_world_cover_transform import (
    EsaWorldCoverTransform,
)
from remote_sensing_core.transforms.ffcv.blow_up import BlowUp
from remote_sensing_core.transforms.ffcv.min_max_scaler import MinMaxScaler
from remote_sensing_core.transforms.ffcv.era5_temperature_s2_transform import (
    Era5TemperatureS2Transform,
)
import torch

from ffcv.transforms import ToTensor, ToDevice
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, IntDecoder
from ffcv.loader import Loader, OrderOption

from tqdm import tqdm

pipeline = {
    "climate_zone": [
        FloatDecoder(),
        MinMaxScaler(maximum_value=29, minimum_value=0, interval_max=1, interval_min=0),
        BlowUp([1, 120, 120]),
        Convert("float32"),
        ToTensor(),
        ToDevice(device=torch.device("cuda")),
    ],
    #'elevation_differ': [FloatDecoder(), ToTensor(), ToDevice(device)],
    "era_5": [
        NDArrayDecoder(),
        Era5TemperatureS2Transform(batch_size=32),
        BlowUp([1, 120, 120]),
        Convert("float32"),
        ToTensor(),
        ToDevice(device=torch.device("cuda")),
    ],
    "esa_worldcover": [
        NDArrayDecoder(),
        EsaWorldCoverTransform(10, 1),
        Convert("int64"),
        ToTensor(),
        ToDevice(device=torch.device("cuda")),
    ],
    "glo_30_dem": [
        NDArrayDecoder(),
        ChannelSelector([0]),
        ToTensor(),
        ToDevice(device=torch.device("cuda")),
    ],
    #'multiclass_numer': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
    "multiclass_one_h": [ToTensor(), ToDevice(device=torch.device("cuda"))],
    "season_s1": [
        FloatDecoder(),
        BlowUp([1, 120, 120]),
        Convert("float32"),
        ToTensor(),
        ToDevice(device=torch.device("cuda")),
    ],
    "season_s2": [
        FloatDecoder(),
        BlowUp([1, 120, 120]),
        Convert("float32"),
        ToTensor(),
        ToDevice(device=torch.device("cuda")),
    ],
    "sentinel_1": [NDArrayDecoder(), ToTensor(), ToDevice(device=torch.device("cuda"))],
    "sentinel_2": [
        NDArrayDecoder(),
        Clipping([0, 10_000]),
        ChannelSelector([7, 3, 2, 1]),
        ToTensor(),
        ToDevice(device=torch.device("cuda")),
    ],
}

# get_data_set_files(wandb.config.dataset_size)[0]
dataloader_train = Loader(
    "/raid/remote_sensing/ben-ge/ffcv/ben-ge-100-train.beton",
    batch_size=32,
    order=OrderOption.RANDOM,
    num_workers=4,
    pipelines=pipeline,
)

# Initialize variables to store the minimum and maximum values
min_value = float("inf")
max_value = float("-inf")

# Iterate over the data loader
for batch in tqdm(dataloader_train):
    # TODO: Change this to the correct index
    tensor = batch[4]

    # Calculate the minimum and maximum values of the tensor
    batch_min = tensor.min().item()
    batch_max = tensor.max().item()

    # Update the overall minimum and maximum values if necessary
    min_value = min(min_value, batch_min)
    max_value = max(max_value, batch_max)

# Save the minimum and maximum values to a file
output_file = "min_max_values_elevation_height_values.txt"
with open(output_file, "w") as file:
    file.write("Minimum value: {}\n".format(min_value))
    file.write("Maximum value: {}".format(max_value))

# Print confirmation message
print("Minimum and maximum values saved to", output_file)
