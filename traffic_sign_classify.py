import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import model_pars as parms

import torchvision.models as models

# class TrafficSignModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Example: use pretrained ResNet18 and adapt final layer
#         self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#         self.model.fc = nn.Sequential(
#             nn.Dropout(0.4),  # Add dropout for regularization
#             nn.Linear(self.model.fc.in_features, num_classes)
#         ) 
    
#     def forward(self, x):
#         return self.model(x)



class TrafficSignModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout_p=0.4):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

def get_transforms():
    train_t = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.2)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_t = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return train_t, val_t

def train_one_epoch(model, loader, optimizer, criterion, device, tag):
    model.train() if tag=='train' else model.eval()
    running_loss = 0.0
    with torch.set_grad_enabled(tag=='train'):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if tag == 'train':
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

    

def training(train_dspath, num_classes, max_epochs, model_output_path,
            patience=5, lr=1e-5, batch_size=64):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_t, val_t = get_transforms()
    full_dataset = datasets.ImageFolder(train_dspath, transform=val_t)
    labels = [y for _, y in full_dataset.imgs]
    train_idx, val_idx = next(splitter.split(full_dataset.imgs, labels))

    train_ds = Subset(
        datasets.ImageFolder(train_dspath, transform=train_t),
        train_idx
    )
    val_ds = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = TrafficSignModel(num_classes=num_classes, pretrained=True, dropout_p=0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    counter = 0

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, 'train')
        val_loss   = train_one_epoch(model, val_loader, optimizer, criterion, device, 'val')
        print(f"Epoch {epoch+1:02d} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        # check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), model_output_path)  # save best
        else:
            counter += 1
            if counter >= patience:
                print(f"→ Early stopping at epoch {epoch+1}, no improvement for {patience} epochs.")
                break

    parms.get_model_size(model)
    
   
    #testing phase
    print("Testing phase...") 
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    
    print(f"Using device: {device}")           
    test_dspath = r'/home/anagha/Documents/MAI/ACV/Portfolio2/Traffic/Data/Test'
    val_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_dataset = datasets.ImageFolder(test_dspath, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Found {len(test_dataset)} test images across {len(test_dataset.classes)} classes")

    # load best weights before testing
    model.load_state_dict(torch.load(model_output_path, map_location=device))
    model.to(device).eval()
    all_preds = []
    all_labels = []
    import time
    start_time = time.time()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    end_time = time.time()
    print(f"Time taken for evaluation: {end_time - start_time:.2f} seconds")

    # FPS calculation
    num_images = len(test_dataset)
    fps = num_images / (end_time - start_time)
    print(f"Frames per second (FPS): {fps:.2f}")

    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    
    

if __name__ == "__main__":
    train_dspath = r'/home/anagha/Documents/MAI/ACV/Portfolio2/Traffic/Data/Train'
    training(
        train_dspath=train_dspath,
        num_classes= 43,
        max_epochs= 100,                
        model_output_path='traffic_sign_model_efficientnet.pth',
        patience=7,                    
        lr=1e-5,
        batch_size=64
    )
