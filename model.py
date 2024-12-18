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
