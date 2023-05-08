from torchvision.transforms import transforms


class Transforms:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                # transforms.ConvertImageDtype(torch.float),
            ]
        )
