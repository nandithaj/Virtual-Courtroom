import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from torchvision.models import VGG16_Weights

# Define paths to the dataset
data_dir = r'C:\Users\reube\OneDrive\Desktop\majorproject\Dataset_Train\train'
test_dir = r'C:\Users\reube\OneDrive\Desktop\majorproject\Dataset_Test\test'

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_data = datasets.ImageFolder(root=data_dir, transform=transform)
val_data = datasets.ImageFolder(root=test_dir, transform=transform)
num_classes = len(train_data.classes)  # Should be 4 (deep, subtentorial, lobar, no hemorrhage)



full_data = datasets.ImageFolder(root=data_dir, transform=transform)
num_classes = len(full_data.classes)  # Should be 4 (deep, subtentorial, lobar, no hemorrhage)
test_data = datasets.ImageFolder(root=r'C:\Users\reube\OneDrive\Desktop\majorproject\Dataset_Test\test', transform=transform)

# Split into training and validation sets (80-20 split)
train_size = int(0.8 * len(full_data))
val_size = len(full_data) - train_size
train_data, val_data = random_split(full_data, [train_size, val_size])

# Data Loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load pre-trained VGG16
class VGG16MultiClass(nn.Module):
    def __init__(self, num_classes):
        super(VGG16MultiClass, self).__init__()
        self.vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Modify the fully connected layer
        num_ftrs = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # Multi-class output
        )

    def forward(self, x):
        return self.vgg16(x)

# Initialize model
model = VGG16MultiClass(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import numpy as np

# Compute Class Weights
class_counts = np.bincount([label for _, label in full_data.samples])
total_samples = sum(class_counts)
class_weights = torch.tensor(total_samples / class_counts, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validation loop
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = 100 * correct_val / total_val
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, '
              f'Val Acc: {val_accuracy:.2f}%')

    # Plot Accuracy & Loss
    plot_training_results(epochs, train_losses, val_losses, train_accuracies, val_accuracies)

# Function to plot training and validation accuracy & loss
def plot_training_results(epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25)
    
    # Save model
    torch.save(model.state_dict(), 'vgg16_30ep_001.pth')
    print("Model saved as vgg16_30ep_001.pth")


     # Generate Confusion Matrix
    model.load_state_dict(torch.load('vgg16_20ep.pth'))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())  # Convert to NumPy
            all_labels.extend(labels.cpu().numpy())

    # Compute Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = full_data.classes  # Extract class names from dataset

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()  # Display the figure without saving

    # Print Classification Report
    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))
