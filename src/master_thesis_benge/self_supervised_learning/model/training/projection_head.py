import torch.nn as nn
from torchvision.models import resnet18
import torch

from master_thesis_benge.self_supervised_learning.model.training.unet_encoder import UNetEncoder

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

        self.projection = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(), 
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x, return_embedding=False):
            embedding = self.backbone(x)
            if return_embedding:
                return embedding
            return self.projection(embedding)


class AddUNetProjection(nn.Module):
    def __init__(self, in_channels, embedding_size, mlp_dim=512):
        super(AddUNetProjection, self).__init__()
        self.encoder = UNetEncoder(in_channels)
        self.projection = nn.Sequential(
            nn.Linear(in_features=1024, out_features=mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x, return_embedding=False):
        encoding = self.encoder(x)
        print(encoding)
        print(encoding.shape)
        input('test')
        print(torch.equal(encoding, encoding.view(encoding.size(0), -1)))
        input('equal')
        encoding = encoding.view(encoding.size(0), -1)
        print(encoding)
        print(encoding.shape)
        input('test')
        embedding = self.projection(encoding)
        
        if return_embedding:
            return embedding
        return self.projection(embedding)