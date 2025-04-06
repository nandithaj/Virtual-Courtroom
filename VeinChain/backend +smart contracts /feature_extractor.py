import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from PIL import Image

# Set random seed for deterministic behavior
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Define device (forcing CPU to avoid floating-point variability)
device = torch.device("cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.LANCZOS),  # Ensure deterministic resizing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize RGB images
])

# Load pre-trained ResNet-50
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Replace the fully connected layer to reduce dimensions to 128
model.fc = nn.Sequential(
    nn.Linear(2048, 128),  # Reduce to 128 dimensions
    nn.ReLU(),             
    nn.BatchNorm1d(128, affine=False)  # Disable affine parameters to keep BN stable
)

model = model.to(device)
model.eval()

# Function to extract feature vector from a PIL Image object
def extract_feature(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))  # Convert NumPy array back to PIL
    image = image.convert("RGB")  # Ensure it's in RGB mode
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(image).cpu().numpy().flatten()

    return feature
