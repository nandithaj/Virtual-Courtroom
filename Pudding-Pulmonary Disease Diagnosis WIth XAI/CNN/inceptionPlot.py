import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from inception import CNN

# Step 1: Load the saved model
model_path = 'inception_model_new.pth'  # Update if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(model_path, map_location=device)
model.eval()  # Set to evaluation mode

# Step 2: Prepare the test dataset
data_dir = "archive/chest-xray-dataset"  # Update the dataset path
test_path = os.path.join(data_dir, 'val')

transform = transforms.Compose([
    transforms.Resize((299, 299)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class_names = test_dataset.classes  # Get class names

# Step 3: Make Predictions
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)  # Get model predictions
        _, preds = torch.max(outputs, 1)  # Convert logits to class indices

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Step 4: Calculate Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

# Print evaluation results
print(f"📊 Model Evaluation Results:")
print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔄 Recall: {recall:.4f}")
print(f"⭐ F1 Score: {f1:.4f}")

# Step 5: Visualization
# 1️⃣ Bar Chart for Accuracy, Precision, Recall, F1-score
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
sns.barplot(x=metrics, y=values, palette='viridis')
plt.ylim(0, 1)  # Scores range from 0 to 1
plt.ylabel("Score")
plt.title("Model Evaluation Metrics")
plt.show()

# 2️⃣ Confusion Matrix as a Heatmap
conf_matrix = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
