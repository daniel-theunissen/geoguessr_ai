import torch
from torch import nn
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class GeoLocViT(nn.Module):
    def __init__(self, n_clusters):
        """
        Vision Transformer for geolocation classification.

        Args:
            n_clusters (int): Number of clusters/classes for classification.
        """
        super(GeoLocViT, self).__init__()
        self.backbone = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        # self.backbone = vit_b_32()
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.heads.head.in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_clusters),  # Output logits for n_clusters
        )

        # Replace the backbone head with an identity layer
        self.backbone.heads.head = nn.Identity()

        self._init_weights()

    def forward(self, x):
        features = self.backbone(x)  # Extract features using ViT backbone
        logits = self.classifier(features)  # Classification head
        return logits

    def _init_weights(self):
        """Initialize the weights of the classifier."""

        def init_function(m):
            if isinstance(m, nn.Linear):
                # Xavier Uniform Initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # Set BatchNorm weights to 1 and biases to 0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Apply initialization to the classifier
        self.classifier.apply(init_function)


class GeoLocResNet(nn.Module):
    def __init__(self, n_clusters):
        """
        ResNet-based model for geolocation classification.

        Args:
            n_clusters (int): Number of clusters/classes for classification.
        """
        super(GeoLocResNet, self).__init__()
        # Load the ResNet backbone
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)

        # Replace the original fully connected layer of ResNet with an identity layer
        self.backbone.fc = nn.Identity()

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),  # 2048 is the output size of ResNet50 backbone
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_clusters),  # Output logits for n_clusters
        )

        # Initialize the weights of the custom classifier
        self._init_weights()

    def forward(self, x):
        # Extract features using the ResNet backbone
        features = self.backbone(x)
        # Pass features through the classifier head
        logits = self.classifier(features)
        return logits

    def _init_weights(self):
        # Custom weight initialization for the classifier
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# class GeoLocViT(nn.Module):
#     def __init__(self, pretrained=True):
#         super(GeoLocViT, self).__init__()
#         # Load a pretrained ViT backbone
#         self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
#         # for param in self.backbone.parameters():
#         #     param.requires_grad = False

#         # Replace the classification head with a regression head for geolocation
#         self.regressor = nn.Sequential(
#             nn.Linear(self.backbone.heads.head.in_features, 512),
#             nn.ReLU(),
#             nn.Linear(512, 2),  # Output: [latitude, longitude]
#         )
#         self.backbone.heads.head = nn.Identity()

#     def forward(self, x):
#         features = self.backbone(x)  # Extract features
#         # return torch.tanh(self.regressor(features))  # Predict [lat, long]
#         return self.regressor(features)  # Predict [lat, long]


import torch


def haversine_loss(pred, target):
    R = 6371  # Earth's radius in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = torch.deg2rad(pred[:, 0]), torch.deg2rad(pred[:, 1])
    lat2, lon2 = torch.deg2rad(target[:, 0]), torch.deg2rad(target[:, 1])

    # Compute differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    # Return mean haversine distance
    return (R * c).mean()


def weighted_real_world_loss(
    pred_norm, target_norm, lat_range=(-90, 90), lon_range=(-180, 180)
):
    """
    Compute a weighted loss in real-world coordinate space.

    Args:
        pred_norm (torch.Tensor): Normalized predicted coordinates (batch_size, 2).
        target_norm (torch.Tensor): Normalized target coordinates (batch_size, 2).
        lat_range (tuple): Latitude range (lat_min, lat_max).
        lon_range (tuple): Longitude range (lon_min, lon_max).

    Returns:
        torch.Tensor: Weighted loss.
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    # Denormalize predictions and targets
    pred_real = torch.stack(
        [
            (pred_norm[:, 0] + 1) / 2 * (lat_max - lat_min) + lat_min,
            (pred_norm[:, 1] + 1) / 2 * (lon_max - lon_min) + lon_min,
        ],
        dim=1,
    )

    target_real = torch.stack(
        [
            (target_norm[:, 0] + 1) / 2 * (lat_max - lat_min) + lat_min,
            (target_norm[:, 1] + 1) / 2 * (lon_max - lon_min) + lon_min,
        ],
        dim=1,
    )

    # Weight loss: latitude and longitude differences
    lat_loss = torch.abs(pred_real[:, 0] - target_real[:, 0])  # Latitude loss
    lon_loss = torch.abs(pred_real[:, 1] - target_real[:, 1])  # Longitude loss

    # Scale longitude by Earth's curvature (cosine factor)
    lat_mean = torch.mean(target_real[:, 0])  # Approximate latitude center
    lon_weight = torch.cos(
        torch.deg2rad(lat_mean)
    )  # Weight by latitude-dependent scale

    loss = torch.mean(lat_loss) + lon_weight * torch.mean(lon_loss)
    return loss


def real_world_loss(pred_norm, target_norm, lat_range=(-90, 90), lon_range=(-180, 180)):
    """
    Compute the loss in real-world coordinate space.

    Args:
        pred_norm (torch.Tensor): Normalized predicted coordinates (batch_size, 2).
        target_norm (torch.Tensor): Normalized target coordinates (batch_size, 2).
        lat_range (tuple): Latitude range (lat_min, lat_max).
        lon_range (tuple): Longitude range (lon_min, lon_max).

    Returns:
        torch.Tensor: Loss computed in real-world space.
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    # Denormalize predictions and targets
    pred_real = torch.stack(
        [
            (pred_norm[:, 0] + 1) / 2 * (lat_max - lat_min) + lat_min,
            (pred_norm[:, 1] + 1) / 2 * (lon_max - lon_min) + lon_min,
        ],
        dim=1,
    )

    target_real = torch.stack(
        [
            (target_norm[:, 0] + 1) / 2 * (lat_max - lat_min) + lat_min,
            (target_norm[:, 1] + 1) / 2 * (lon_max - lon_min) + lon_min,
        ],
        dim=1,
    )

    # Compute L2 loss in real-world space
    loss = torch.mean(
        torch.norm(pred_real - target_real, dim=1)
    )  # Mean Euclidean distance
    return loss
