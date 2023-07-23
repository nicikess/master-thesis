import torch.nn as nn
from torchvision.models import resnet18
import torch

from master_thesis_benge.self_supervised_learning.model.training.unet_encoder import UNetEncoder

def create_projection(input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(in_features=input_dim, out_features=output_dim),
            nn.BatchNorm1d(output_dim),
        )

class AddResNetProjection(nn.Module):
    def __init__(self, in_channels, embedding_size, mlp_dim=512):
        super(AddResNetProjection, self).__init__()
        self.backbone = resnet18(weights=None, num_classes=embedding_size)
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        mlp_dim = self.backbone.fc.in_features
        print("Dim MLP input:", mlp_dim)
        self.backbone.fc = nn.Identity()

        self.projection = create_projection(mlp_dim, embedding_size)

    def forward(self, x, return_embedding=False):
            embedding = self.backbone(x)
            if return_embedding:
                return embedding
            return self.projection(embedding)


class AddUNetProjection(nn.Module):
    def __init__(self, in_channels, embedding_size, mlp_dim=512):
        super(AddUNetProjection, self).__init__()
        self.encoder = UNetEncoder(in_channels)
        self.projection = create_projection(mlp_dim, embedding_size)


    def forward(self, x, return_embedding=False):
        embedding = self.encoder(x)
        pooled_embedding = nn.AdaptiveAvgPool2d(1)(embedding).view(128,512)        
        if return_embedding:
            return embedding
        return self.projection(pooled_embedding)
