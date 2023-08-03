import torch
from models.resnet import UniResNet
from models.dual_resnet import DualResNet


def init_resnet(in_channels_1, number_of_classes):
    # Create the model
    model = UniResNet(in_channels_1=in_channels_1, number_of_classes=number_of_classes)

    print(model)

    # Load the state_dict
    state_dict = torch.load(
        "models/weights/state_dict_sentinel2.pt", map_location=torch.device("cpu")
    )

    # Load the state_dict into the model
    model.load_state_dict(state_dict)

    # Example input tensor and forward pass
    x = torch.randn(1, 4, 120, 120)
    output = model(x)

    print("Example output for ResNet with one modality: ", output)


def init_dual_resnet(in_channels_1, in_channels_2, number_of_classes):
    # Create the model
    model = DualResNet(
        in_channels_1=in_channels_1,
        in_channels_2=in_channels_2,
        number_of_classes=number_of_classes,
    )

    # Load the state_dict
    state_dict = torch.load(
        "models/weights/state_dict_sentinel2-sentinel1.pt",
        map_location=torch.device("cpu"),
    )

    # Load the state_dict into the model
    model.load_state_dict(state_dict)

    # Example usage
    x1 = torch.randn(1, 4, 120, 120)  # Example S2 tensor
    x2 = torch.randn(1, 2, 120, 120)  # Example S1 tensor

    output = model(x1, x2)  # Forward pass

    print("Example output for ResNet with two modalities: \n", output)


if __name__ == "__main__":
    # Set the number of input channels for the first modality
    in_channels_1 = 4

    # Set the number of input channels for the second modality
    in_channels_2 = 2

    # Set the number of classes
    number_of_classes = 11

    # Initialize the models and make example forward passes
    init_resnet(in_channels_1=in_channels_1, number_of_classes=number_of_classes)
    # init_dual_resnet(in_channels_1=in_channels_1, in_channels_2=in_channels_2, number_of_classes=number_of_classes)
