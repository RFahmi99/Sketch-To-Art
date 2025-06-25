"""
Sketch-to-Image Inference Script
--------------------------------
Converts input sketches to realistic images using a trained UNet model.

Usage:
1. Load pre-trained model
2. Preprocess input image
3. Generate output image
4. Save result

Note: Ensure model weights match the architecture
"""

import torch
import torchvision.utils as vutils
import albumentations as A
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2
from models.unet import UNet

# Initialize UNet model
model = UNet()

# Load pre-trained weights
model_path = "./models/weights/model.pth"  # Path to your checkpoint file
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

# Load and preprocess input image
image = Image.open("path/to/inference/image").convert("RGB")
image = np.array(image)

# Inference transformations (no augmentations)
inference_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
], additional_targets={'mask': 'image'})

# Apply transformations and add batch dimension
image_tensor = inference_transform(image=image)["image"]
input_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

# Model inference
with torch.no_grad():
    output_tensor = model(input_tensor)

# Post-process and save output
output_tensor = output_tensor.squeeze(0).detach().cpu()
output_image = vutils.make_grid(output_tensor, normalize=True)
vutils.save_image(output_image, "outputs/generated_art.png")