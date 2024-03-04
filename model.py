import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import os


class BreastCancerDataset(Dataset):
        def __init__(self, images_dir, labels_dir, transform=None):
            self.images_dir = images_dir
            self.labels_dir = labels_dir
            self.transform = transform
            self.images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_dir) for f in filenames if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]
            self.labels = [os.path.join(dp, f) for dp, dn, filenames in os.walk(labels_dir) for f in filenames if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_name = self.images[idx]
            label_name = self.labels[idx]
            image = Image.open(img_name).convert('L')  # Convert to grayscale
            label = Image.open(label_name).convert('L')
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)
            return image, label

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((skip, x), dim=1)  # Skip connection
        x = self.conv(x)
        return x

class ViT(nn.Module):
    def __init__(self, in_channels, img_dim, patch_dim):
        super(ViT, self).__init__()
        # Assuming img_dim and patch_dim are powers of 2
        num_patches = (img_dim // patch_dim) ** 2
        self.patch_dim = patch_dim
        self.d_model = in_channels * patch_dim ** 2

        self.flatten = nn.Flatten(2)
        self.fc = nn.Linear(self.d_model, self.d_model)
        self.transformer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dropout=0.1)

    def forward(self, x):
        # Create patches
        x = x.unfold(2, self.patch_dim, self.patch_dim).unfold(3, self.patch_dim, self.patch_dim)
        x = self.flatten(x)
        x = self.fc(x)
        # Transformer
        x = self.transformer(x)
        return x

class TransUnet(nn.Module):
    def __init__(self, in_channels, out_channels, img_dim, patch_dim):
        super(TransUnet, self).__init__()

        # Encoder using ResNet
        self.resnet = resnet.resnet50(weights=resnet.ResNet50_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool
        self.encoder0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3

        # Vision Transformer
        self.vit = ViT(1024, img_dim // 16, patch_dim)  # 1024 is the output channels from resnet.layer3

        # Decoder
        self.upconv4 = UpConv(1024, 512)
        self.upconv3 = UpConv(1024, 256)
        self.upconv2 = UpConv(512, 128)
        self.upconv1 = UpConv(256, 64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        # Encoder
        skip0 = self.encoder0(x)
        skip1 = self.encoder1(skip0)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        x = self.encoder4(skip3)

        # ViT
        x = self.vit(x)

        # Decoder
        x = self.upconv4(x, skip3)
        x = self.upconv3(x, skip2)
        x = self.upconv2(x, skip1)
        x = self.upconv1(x, skip0)  # Corrected skip connection

        x = self.final_conv(x)

        return x
