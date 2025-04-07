import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from cnn import CNNModel
from densenet import DenseNetMultiClass
from resnet import ResNetMultiClass
from vgg16 import VGG16MultiClass
from efficient import EfficientNetMultiClass
import torch.nn.functional as F
import time
from cnn_gradcam import GradCAM
import cv2
import numpy as np

class_names = ['Deep', 'Lobar','No Hemorrhage','SubTentorial']
STATIC_FOLDER = r"C:\Users\reube\OneDrive\Desktop\majorproject\ICH_detection\static"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# cnn_model='cnn_model_30ep_001.pth'
cnn_model='efficientnet_20_lr0001ep.pth'
# densenet_model='densenet_multi_30ep_00001.pth'
densenet_model='densenet_multi_30ep_32b_lr5.pth'
resnet_model='resnet50_30_lr00005ep.pth'
# resnet_model='resnet50_20ep.pth'
vgg_model='vgg16_20ep.pth'
# vgg_model='vgg16_30ep_001.pth'


def predict_image_cnn(image_path):
    # model = CNNModel(num_classes=len(class_names))
    model=EfficientNetMultiClass(num_classes=len(class_names))
    model.load_state_dict(torch.load(cnn_model, map_location=device,weights_only=True))
    model.to(device)
    model.eval()
    image = Image.open(image_path).convert("RGB")  
    image = transform1(image).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        # print(predicted_class_idx)
        predicted_class = class_names[predicted_class_idx]

    print(f"Predicted Class(EFF): {predicted_class}   ")
    return predicted_class

def predict_image_densenet(image_path):
    model = DenseNetMultiClass(num_classes=len(class_names))
    model.load_state_dict(torch.load(densenet_model, map_location=device,weights_only=True))
    model.to(device)
    model.eval()  
    image = Image.open(image_path).convert("RGB")  
    image = transform(image).unsqueeze(0).to(device)  

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1) 
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = class_names[predicted_class_idx]

    print(f"Predicted Class(Densenet): {predicted_class}")
    return predicted_class

def predict_image_resenet(image_path):
    model = ResNetMultiClass(num_classes=len(class_names))
    model.load_state_dict(torch.load(resnet_model, map_location=device,weights_only=True))
    model.to(device)
    model.eval()  
    image = Image.open(image_path).convert("RGB")  
    image = transform(image).unsqueeze(0).to(device)  
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1) 
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = class_names[predicted_class_idx]

    print(f"Predicted Class(Resnet): {predicted_class}")
    return predicted_class

def predict_image_vgg16(image_path):
    model = VGG16MultiClass(num_classes=len(class_names))
    model.load_state_dict(torch.load(vgg_model, map_location=device,weights_only=True))
    model.to(device)
    model.eval()  # Set to evaluation mode
    image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)      
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = class_names[predicted_class_idx]

    print(f"Predicted Class(VGG): {predicted_class}")
    return predicted_class

def gradcam_cnn(image_path):
    # model=CNNModel(num_classes=4)
    model=EfficientNetMultiClass(num_classes=4)
    model.load_state_dict(torch.load(cnn_model, map_location=torch.device('cpu'),weights_only=True))
    model.eval()

    # gradcam = GradCAM(model, model.conv_layers[-1])
    gradcam = GradCAM(model, model.efficientnet.features[-1])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform1(image)
    heatmap = gradcam.generate_heatmap(image_tensor)
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (128, 128))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    result = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)


    output_path = os.path.join(STATIC_FOLDER, "cnn_gradcam.jpg")
    cv2.imwrite(output_path, result)

def gradcam_densenet(image_path):

    model=DenseNetMultiClass(num_classes=4)
    model.load_state_dict(torch.load(densenet_model, map_location=torch.device('cpu'),weights_only=True))
    model.eval()

    gradcam = GradCAM(model,model.densenet.features[-3])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    heatmap = gradcam.generate_heatmap(image_tensor)
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (128, 128))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    result = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)


    output_path = os.path.join(STATIC_FOLDER, "densenet_gradcam.jpg")
    cv2.imwrite(output_path, result)

def gradcam_resnet(image_path):
    model=ResNetMultiClass(num_classes=4)
    model.load_state_dict(torch.load(resnet_model, map_location=torch.device('cpu'),weights_only=True))
    model.eval()

    gradcam = GradCAM(model,model.resnet.layer4[-1].conv1)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    heatmap = gradcam.generate_heatmap(image_tensor)
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (128, 128))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    result = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)


    output_path = os.path.join(STATIC_FOLDER, "resnet_gradcam.jpg")
    cv2.imwrite(output_path, result)

def gradcam_vgg16(image_path):
    model=VGG16MultiClass(num_classes=4)
    model.load_state_dict(torch.load(vgg_model, map_location=torch.device('cpu'),weights_only=True))
    model.eval()

    gradcam = GradCAM(model, model.vgg16.features[-1])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    heatmap = gradcam.generate_heatmap(image_tensor)
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (128, 128))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    result = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)


    output_path = os.path.join(STATIC_FOLDER, "vgg16_gradcam.jpg")
    cv2.imwrite(output_path, result)



if __name__ == "__main__":
    before_time = time.time()

    img='uploads\\lobar2.jpg'

    predict_image_cnn(img)
    predict_image_densenet(img)
    predict_image_resenet(img)
    predict_image_vgg16(img)
    gradcam_cnn(img)
    gradcam_densenet(img)
    gradcam_resnet(img)
    gradcam_vgg16(img)

    print(time.time()-before_time)