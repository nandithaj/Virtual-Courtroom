import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activation = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor.requires_grad_()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        gradcam = torch.sum(weights * self.activation, dim=1).squeeze(0)
        gradcam = torch.relu(gradcam).cpu().numpy()

        gradcam = cv2.resize(gradcam, (224, 224))
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
        return gradcam


def visualize_gradcam(heatmap, original_image, output_path="gradcam_output.jpg"):
    original_image = original_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    original_image = np.clip(original_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap / 255.0 + original_image
    overlay = overlay / overlay.max()
    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig(output_path)
    plt.show()
    print(f"GradCAM heatmap saved as {output_path}")
