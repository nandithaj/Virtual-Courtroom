from flask import Flask, request, redirect, flash, send_from_directory, render_template
import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import torch.nn.functional as F
from lime.lime_image import LimeImageExplainer
from sklearn.cluster import KMeans
import shap
import matplotlib
matplotlib.use("Agg")   # To prevent matplot from running on different threads
import matplotlib.pyplot as plt
from skimage.segmentation import slic
import timm
from torchvision.models import ResNet50_Weights

UPLOAD_FOLDER = './uploads'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key="Hello World"

# Define the CNN Model Architecture
class ConvNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        x = self.pool7(self.relu7(self.bn7(self.conv7(x))))
        x = self.dropout(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the Models
num_classes = 2
convnet_model = ConvNet(num_classes)
convnet_model.load_state_dict(torch.load("./best_model_cnn_3.pth", weights_only=True))
convnet_model.eval()

inception_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
inception_model.fc = nn.Linear(2048, num_classes)
inception_model.AuxLogits.fc = nn.Linear(768, num_classes)
inception_model.load_state_dict(torch.load("./best_inception_model3.pth", map_location=device, weights_only=True))
inception_model = inception_model.to(device)
inception_model.eval()

xception_model = timm.create_model('xception', pretrained=False)
xception_model.last_linear = torch.nn.Linear(2048, num_classes)  
xception_model.load_state_dict(torch.load("best_xception_model3.pth", map_location=device, weights_only=True))
xception_model = xception_model.to(device)
xception_model.eval()

resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model.load_state_dict(torch.load("best_resnet50_model3.pth", map_location=device, weights_only=True))
resnet_model = resnet_model.to(device)
resnet_model.eval()  # Set model to evaluation mode

# Define Image Transformations
transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def get_bounding_box_from_heatmap(heatmap, threshold=0.75):
    heatmap = normalize_heatmap(heatmap)
    mask = heatmap > np.percentile(heatmap, threshold * 100)
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return (0,0,10,10)
    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)
    return (x_min, y_min, x_max, y_max)

def compute_iou(boxA, boxB=(50, 50, 175, 175)):
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-8)

def normalize_heatmap(heatmap):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return np.uint8(255 * heatmap)

def lime(file_path, model):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  
    
    explainer = LimeImageExplainer()
    #Perturbation function
    def predict_fn(perturbed_images):
        # Convert perturbed images to tensors and normalize
        perturbed_tensors = [torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0 for img in perturbed_images]
        perturbed_tensors = torch.stack(perturbed_tensors)  # Stack into batch format

        with torch.no_grad():
            predictions = model(perturbed_tensors)  # Get model predictions
        return predictions.cpu().numpy()  # Convert to NumPy
    
    def segmentation_fn(image):
        return slic(image, n_segments=50, compactness=10, sigma=1)

    explanation = explainer.explain_instance(image, predict_fn, top_labels=1, hide_color=0, num_samples=1000, segmentation_fn=segmentation_fn)
    return normalize_heatmap(explanation.get_image_and_mask(explanation.top_labels[0])[1])

def shap_gen(file_path, model):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224)) 
     
    background = torch.randn((10, 3, 224, 224)) # Create dataset to estimate feature importance
    image_tensor = transformer(image).unsqueeze(0)
    
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(image_tensor)
    
    shap_values = np.array(shap_values[0])  # Use the SHAP values for the first class
    input_image = image_tensor.cpu().numpy().transpose(0, 2, 3, 1)[0]  # Convert to NHWC format (Numpy Height Width Channels) from NCHW
    shap_image = shap_values[0]
    
    # Sum across the 3 channels to match image dimensions
    shap_sum = np.sum(shap_image, axis=-1)
    
    shap_sum = (shap_sum - np.min(shap_sum)) / (np.max(shap_sum) - np.min(shap_sum))  # Normalize to [0, 1]
    shap_sum = shap_sum * 2 - 1  # Scale to [-1, 1] for a balanced red-blue colormap
    shap_sum[np.abs(shap_sum) < 0.05] = 0  # Filter out low SHAP values below threshold

    grayscale_image = np.dot(input_image[..., :3], [0.2989, 0.587, 0.114])  # Convert to grayscale
    grayscale_image = np.stack([grayscale_image] * 3, axis=-1)  # Convert back to 3 channels for visualization
    # Lighten the input image for better visibility
    lightened_image = np.clip(input_image * 0.7 + 0.3, 0, 1)

    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"shap_explanation_{name}.png"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)

    # Plot and save the SHAP explanation
    plt.figure(figsize=(8, 6))
    plt.imshow(lightened_image)
    plt.imshow(shap_sum, cmap='bwr', alpha=0.6)  # Increased alpha for better visibility
    plt.colorbar(label="Normalized SHAP value")
    plt.title("SHAP Explanation (Enhanced Contrast)")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return output_path

def gradcam(file_path, target_layer, model):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  
    
    image_tensor = transformer(image).unsqueeze(0)
    activations, gradients = {}, {}

    #Hooks for analyzing the thought process
    def forward_hook(module, input, output):
        activations['value'] = output   # Contains feature maps

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    handle1 = target_layer.register_forward_hook(forward_hook)  # Captures feature maps
    handle2 = target_layer.register_full_backward_hook(backward_hook)   # Captures gradients
    
    output = model(image_tensor)
    target_class = torch.argmax(output, dim=1).item()
    output[:, target_class].backward()  # Computes gradients w.r.t. target class
    
    # Prevent memory leaks
    handle1.remove()
    handle2.remove()
    
    grad = gradients['value'].mean(dim=[2, 3], keepdim=True)
    cam = F.relu(grad * activations['value']).sum(dim=1).squeeze().detach().cpu().numpy()   # Ensure positive contributions
    return target_class, normalize_heatmap(cv2.resize(cam, (224, 224)))

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def gen_heatmap(file, img_path, heatmap, xai):
    # Open and resize the original image
    img = Image.open(img_path).resize((224, 224))  
    img_np = np.array(img) 
    
    # Ensure heatmap has 3 channels and matches image size
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) 
    heatmap = cv2.resize(heatmap, img_np.shape[:2][::-1])
    
    # Create overlay
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    
    # Save and render the result
    if xai == "gradcam":
        path = os.path.join(app.config['UPLOAD_FOLDER'], f"gradcam_{file.filename}")
    elif xai == "lime":
        path = os.path.join(app.config['UPLOAD_FOLDER'], f"lime_{file.filename}")
        
    Image.fromarray(overlay).save(path)
    
    return path

@app.route('/upload', methods=['POST'])
def upload():
        
    if 'file1' not in request.files or request.files['file1'].filename == '':
        flash("Please upload a file!", "warning")
        return redirect('/')

    file = request.files['file1']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file_path = file_path.replace("\\", "/")
    file.save(file_path)
    print(file_path)

    model_type = request.form.get("model_type", "convnet")    # Get mdoel type from form, convnet is default
    
    if model_type == "convnet":
        try:         
            target_layer = convnet_model.conv7  # Use the final layer
            predicted_class, gradcam_heatmap = gradcam(file_path, target_layer, convnet_model)     
            lime_heatmap = lime(file_path, convnet_model)
            shap_path = shap_gen(file_path, convnet_model)
            
            # lime_box = get_bounding_box_from_heatmap(lime_heatmap)
            # #shap_box = get_bounding_box_from_heatmap(shap_heatmap)
            # gradcam_box = get_bounding_box_from_heatmap(gradcam_heatmap)
            # l_iou = compute_iou(lime_box)
            # g_iou = compute_iou(gradcam_box)
            # print("LIME Box:", lime_box)
            # print("Grad-CAM Box:", gradcam_box)
            # print(f"{g_iou:.3f}") 
            # print(f"{l_iou:.3f}") 
            
            gradcam_path = gen_heatmap(file, file_path, gradcam_heatmap, "gradcam")
            lime_path = gen_heatmap(file, file_path, lime_heatmap, "lime")
            return render_template('index.html', 
                                prediction=f'Predicted class: {predicted_class}', 
                                gradcam_image=gradcam_path,
                                lime_image=lime_path,
                                shap_image=shap_path,
                                file_path=file_path)

        except Exception as e:
            return f"An error occurred for CNN: {str(e)}"
    elif model_type == "inception":
        try:

            target_layer = inception_model.Mixed_7c  # Final convolutional block
            predicted_class, gradcam_heatmap = gradcam(file_path, target_layer, inception_model)
            lime_heatmap = lime(file_path, inception_model)
            shap_path = shap_gen(file_path, inception_model)

            gradcam_path = gen_heatmap(file, file_path, gradcam_heatmap, "gradcam")
            lime_path = gen_heatmap(file, file_path, lime_heatmap, "lime")
            
            return render_template('inception.html', 
                                prediction=f'Predicted class: {predicted_class}', 
                                gradcam_image=gradcam_path,
                                lime_image=lime_path,
                                shap_image=shap_path,
                                file_path=file_path)
        except Exception as e:
            return f"An error occurred for Inception: {str(e)}"
    elif model_type == "xception":
        try:

            target_layer = xception_model.conv4   # Final convolutional block
            predicted_class, gradcam_heatmap = gradcam(file_path, target_layer, xception_model)
            lime_heatmap = lime(file_path, xception_model)
            
            gradcam_path = gen_heatmap(file, file_path, gradcam_heatmap, "gradcam")
            lime_path = gen_heatmap(file, file_path, lime_heatmap, "lime")
            
            return render_template('xception.html', 
                                prediction=f'Predicted class: {predicted_class}', 
                                gradcam_image=gradcam_path,
                                lime_image=lime_path,
                                file_path=file_path)
        except Exception as e:
            return f"An error occurred for Xception: {str(e)}"
    else:
        try:
            
            target_layer = resnet_model.layer4[-1]   # Final convolutional block
            predicted_class, gradcam_heatmap = gradcam(file_path, target_layer, resnet_model)
            lime_heatmap = lime(file_path, resnet_model)
            shap_path = shap_gen(file_path, resnet_model)
            
            gradcam_path = gen_heatmap(file, file_path, gradcam_heatmap, "gradcam")
            lime_path = gen_heatmap(file, file_path, lime_heatmap, "lime")
            
            return render_template('resnet.html', 
                                prediction=f'Predicted class: {predicted_class}', 
                                gradcam_image=gradcam_path,
                                lime_image=lime_path,
                                shap_image=shap_path,
                                file_path=file_path)
        except Exception as e:
            return f"An error occurred for ResNET50: {str(e)}"
    
@app.route('/inception')
def inc():
    return render_template('inception.html')

@app.route('/xception')
def xcp():
    return render_template('xception.html')

@app.route('/resnet')
def res():
    return render_template('resnet.html')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
