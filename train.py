import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def iou_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    total = (pred + target).sum(dim=2).sum(dim=2)
    union = total - intersection 

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU.mean()


import torch
import torch.optim as optim

def train_model(model, train_loader, num_epochs=25, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print("hii")
        model.train()  
        train_loss = 0.0

        for images, true_masks in train_loader:
            images, true_masks = images.to(device), true_masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = dice_loss(outputs, true_masks) + iou_loss(outputs, true_masks)
            print("loss",loss)
            train_loss += loss.item()


            loss.backward()
            optimizer.step()

        # Calculate average loss over an epoch
        train_loss /= len(train_loader)

        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}')


    torch.save(model.state_dict(), 'transunet_model.pth')
    print("model saved")

    return model
