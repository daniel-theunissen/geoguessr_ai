import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os, os.path
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class GeoLocationDataset(Dataset):
    def __init__(self, data_dir, model_type):
        self.data_dir = data_dir
        self.targets = np.load(os.path.join(data_dir, "targets.npy"), allow_pickle=True)
        self.model_type = model_type

    def __len__(self):
        return len(os.listdir(self.data_dir)) - 1

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, f"street_view_{idx}.jpg")

        if self.model_type == "vit":
            transform = ViT_B_32_Weights.DEFAULT.transforms()
        elif self.model_type == "resnet":
            transform = ResNet50_Weights.DEFAULT.transforms()
        elif self.model_type == "wideres":
            transform = Wide_ResNet50_2_Weights.DEFAULT.transforms()

        # transform = transforms.Compose([
        #         # Resize the image
        #         transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        #         # Center crop to 224x224
        #         transforms.CenterCrop(224),
        #         # Convert to tensor and scale to [0.0, 1.0]
        #         transforms.ToTensor(),
        #         # Normalize using mean and std
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ])

        img = pil_loader(data_path)
        data = transform(img)

        target = torch.tensor(self.targets[idx], dtype=torch.long)

        return data, target


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
