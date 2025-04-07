import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
from torch.utils.data import DataLoader
import os
from tqdm import tqdm  # Import tqdm for progress bars

# Device configuration
device = torch.device('cuda')
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.get_device_name(0))  # GPU name
print(device)

# Directories
train_dir = "archive/chest-xray-dataset/train"
val_dir = "archive/chest-xray-dataset/val"

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 50
num_classes = len(os.listdir(train_dir))

# Data Transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),         # Convert images to PyTorch tensors
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize
])

# Datasets and Dataloaders
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Custom CNN Model
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Add BatchNorm
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Add BatchNorm
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # Adjust input size
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        # Save conv output for Grad-CAM use
        if not self.training:
            x.requires_grad_()  # Force requires_grad to True
            x.retain_grad()     # Now you can safely retain
        self.last_conv_output = x

        #print("Shape before flattening:", x.shape)  # Debugging
        x = x.view(x.size(0), -1)  # Flatten
        #print("Shape after flattening:", x.shape)  # Debugging
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CustomCNN(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with progress bars
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_accuracy = 0.0  # To save the best model based on validation accuracy
    save_path = "customModelCOVID.pth"
    save_dict_path = "customModelDictCOVID.pth"
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop with progress bar
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        train_progress = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_progress.set_postfix(loss=running_loss / len(train_loader))
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}")
        
        # Validation loop with progress bar
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_progress = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, labels in val_progress:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_progress.set_postfix(loss=val_loss / len(val_loader))
        
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Save the model if validation accuracy improves
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model, save_path)  # Save entire model
            torch.save(model.state_dict(), save_dict_path)  # Save model's state dict
            print(f"Model saved to {save_path} with accuracy: {val_accuracy:.2f}%")
    
    torch.save(model, "customModelCOVIDFinal.pth")  # Save entire model
    torch.save(model.state_dict(), "customModelDictCOVIDFinal.pth")  # Save model's state dict

# Call the training function
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)


images, _ = next(iter(val_loader))  # Get one batch of validation images
torch.save(images, "val_images.pt")  # Save them for SHAP
print("Saved validation images for SHAP analysis.")