import torch
import torch.nn as nn
import os
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm  # Progress bar library

# Step 1: Data Preparation
class LungXrayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.dataset = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def prepare_data(data_dir,batch_size=8):
    transform1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomCrop(150, padding=4),
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        # transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform2 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(data_dir,'train')
    val_path = os.path.join(data_dir,'val')

    train_dataset = LungXrayDataset(train_path, transform=transform1)
    val_dataset = LungXrayDataset(val_path, transform=transform2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(train_dataset.dataset.classes)

# Step 2: Build the R-CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
       
        # Freeze base model layers
        for name, param in self.base_model.named_parameters():
            if "layer4" in name or "fc" in name:  # Fine-tune layer4 and fc layers
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Replace the fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


# Step 3: Train the Model
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.0001):
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(
        [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr": 1e-4},
        ]
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
       
        # Add progress bar for training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=(running_loss / len(train_loader)))

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Add progress bar for validation
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = correct / total
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model, 'resnet_model_new.pth')
            torch.save(model.state_dict(), 'resnet_dict_new.pth')
            print(f"Model with accuracy {100 * correct / total:.2f} and saved as 'resnet_model.pth'.")
    
    torch.save(model, 'resnet_model_final.pth')
    torch.save(model.state_dict(), 'resnet_dict_final.pth')

# Step 4: Main Function to Run the Process
def main(data_dir):
    train_loader, val_loader, num_classes = prepare_data(data_dir)
    model = CNN(num_classes)
    print("Checking trainable parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    data_dir = "archive/chest-xray-dataset"  # Replace with the actual path to your dataset
    main(data_dir)