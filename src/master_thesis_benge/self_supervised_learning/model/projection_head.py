import torch.nn as nn
from torchvision.models import resnet18


class AddProjection(nn.Module):
    def __init__(self, in_channels, embedding_size, mlp_dim=512):
        super(AddProjection, self).__init__()
        self.backbone = resnet18(weights=None, num_classes=embedding_size)
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        mlp_dim = self.backbone.fc.in_features
        print("Dim MLP input:", mlp_dim)
        self.backbone.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(), 
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x_s2=None, x_s1=None, return_embedding=False):
        if x_s2 is not None:
            embedding = self.backbone(x_s2)
            if return_embedding:
                return embedding
            return self.projection(embedding)
        elif x_s1 is not None:
            embedding = self.backbone(x_s1)
            if return_embedding:
                return embedding
            return self.projection(embedding)
        else:
            raise ValueError("Either x_s2 or x_s1 must be provided.")
