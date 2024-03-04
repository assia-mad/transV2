import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import BreastCancerDataset, TransUnet
from train import train_model
import torch.nn as nn

def main():
    images_dir = 'C:/Users/windows/Desktop/Datasets/BC_dataset/seg/train/images'
    labels_dir = 'C:/Users/windows/Desktop/Datasets/BC_dataset/seg/train/labels'
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    seg_dataset = BreastCancerDataset(images_dir=images_dir, labels_dir=labels_dir, transform=transform)
    seg_loader = DataLoader(seg_dataset, batch_size=16, shuffle=True)

    # Model setup
    model = TransUnet(in_channels=1, out_channels=1, img_dim=128, patch_dim=16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    # Training parameters
    num_epochs = 10

    # Start training
    print("we will start training")
    train_model(model, seg_loader, num_epochs,learning_rate=0.0001)

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
