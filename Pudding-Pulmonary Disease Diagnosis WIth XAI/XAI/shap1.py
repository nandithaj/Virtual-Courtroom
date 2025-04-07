import torch
import shap
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from newmodel import CNN  # Ensure CNN is properly defined

class SHAPExplainer:
    def __init__(self, model_path="COVID_dict_new.pth", num_classes=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CNN model
        self.model = CNN(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        # Define preprocessing
        self.transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def explain(self, image_path, save_path="shap_output.png"):
        """Generates SHAP explanations for a given image and saves the visualization."""
        image = Image.open(image_path).convert("RGB")  
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Define SHAP explainer
        background = torch.randn(10, 3, 224, 224).to(self.device)
        explainer = shap.GradientExplainer(self.model, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(image.clone().detach())
        shap_numpy = np.array(shap_values)[0]

        # Ensure SHAP values are 2D (height, width)
        if shap_numpy.ndim == 4:  # (batch, channels, height, width)
            # Debugging: Print shape before processing
            print("SHAP shape before processing:", shap_numpy.shape)

            # Ensure SHAP values are 2D by averaging over channels & classes
            shap_numpy = np.mean(shap_numpy, axis=(0, -1))  # Average over channel & class dimensions

            # Debugging: Print new shape
            print("SHAP shape after processing:", shap_numpy.shape)  # Should be (224, 224)
            
        if shap_numpy.ndim == 3:  # (channels, height, width)
            shap_numpy = np.mean(shap_numpy, axis=0)  # Average over channels

        # Normalize SHAP values
        shap_numpy = (shap_numpy - shap_numpy.min()) / (shap_numpy.max() - shap_numpy.min() + 1e-8)
        shap_uint8 = (shap_numpy * 255).astype(np.uint8)

        # Ensure the shape is 2D before applying color map
        if shap_uint8.ndim == 3:
            shap_uint8 = np.mean(shap_uint8, axis=0).astype(np.uint8)

        # Apply color map and resize
        shap_colored = cv2.applyColorMap(shap_uint8, cv2.COLORMAP_TURBO)
        shap_colored = cv2.resize(shap_colored, (224, 224))

        # Load original image
        original_image = Image.open(image_path).convert("RGB").resize((224, 224))
        original_np = np.array(original_image)

        # Ensure grayscale images are converted to RGB
        if original_np.ndim == 2:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)

        # Adjust blending to make SHAP overlay clearer
        overlay = cv2.addWeighted(original_np, 0.4, shap_colored, 0.7, 0)  # Increased heatmap weight

        # Plot SHAP heatmap
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title("SHAP Explanation for CNN Model")
        plt.show()

        # Save the SHAP output
        plt.savefig(save_path)
        plt.close()

        return save_path