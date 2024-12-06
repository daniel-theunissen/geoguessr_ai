import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class GeoLocViT(nn.Module):
    def __init__(self, pretrained=True):
        super(GeoLocViT, self).__init__()
        # Load a pretrained ViT backbone
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # Replace the classification head with a regression head for geolocation
        self.regressor = nn.Sequential(
            nn.Linear(self.backbone.heads.head.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # Output: [latitude, longitude]
        )
        self.backbone.heads.head = nn.Identity()

    def forward(self, x):
        features = self.backbone(x)  # Extract features
        #return torch.tanh(self.regressor(features))  # Predict [lat, long]
        return self.regressor(features)  # Predict [lat, long]

def haversine_loss(pred, target):
    R = 6371  # Earth's radius in kilometers
    # lat1, lon1 = torch.deg2rad(pred[:, 0]), torch.deg2rad(pred[:, 1])
    # lat2, lon2 = torch.deg2rad(target[:, 0]), torch.deg2rad(target[:, 1])
    lat1, lon1 = torch.deg2rad(pred[0]), torch.deg2rad(pred[1])
    lat2, lon2 = torch.deg2rad(target[0]), torch.deg2rad(target[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return (R * c).mean()  # Mean haversine distance

def weighted_real_world_loss(pred_norm, target_norm, lat_range=(-90, 90), lon_range=(-180, 180)):
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
    pred_real = torch.stack([
        (pred_norm[:, 0] + 1) / 2 * (lat_max - lat_min) + lat_min,
        (pred_norm[:, 1] + 1) / 2 * (lon_max - lon_min) + lon_min
    ], dim=1)

    target_real = torch.stack([
        (target_norm[:, 0] + 1) / 2 * (lat_max - lat_min) + lat_min,
        (target_norm[:, 1] + 1) / 2 * (lon_max - lon_min) + lon_min
    ], dim=1)

    # Weight loss: latitude and longitude differences
    lat_loss = torch.abs(pred_real[:, 0] - target_real[:, 0])  # Latitude loss
    lon_loss = torch.abs(pred_real[:, 1] - target_real[:, 1])  # Longitude loss

    # Scale longitude by Earth's curvature (cosine factor)
    lat_mean = torch.mean(target_real[:, 0])  # Approximate latitude center
    lon_weight = torch.cos(torch.deg2rad(lat_mean))  # Weight by latitude-dependent scale

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
    pred_real = torch.stack([
        (pred_norm[:, 0] + 1) / 2 * (lat_max - lat_min) + lat_min,
        (pred_norm[:, 1] + 1) / 2 * (lon_max - lon_min) + lon_min
    ], dim=1)

    target_real = torch.stack([
        (target_norm[:, 0] + 1) / 2 * (lat_max - lat_min) + lat_min,
        (target_norm[:, 1] + 1) / 2 * (lon_max - lon_min) + lon_min
    ], dim=1)

    # Compute L2 loss in real-world space
    loss = torch.mean(torch.norm(pred_real - target_real, dim=1))  # Mean Euclidean distance
    return loss

