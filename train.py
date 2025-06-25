"""
UNet Training Script
--------------------
Trains attention-based UNet for sketch-to-image translation.

Features:
- Data augmentation
- Learning rate scheduling
- Checkpoint saving
- Multi-GPU support
"""

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import trainModel
from models.unet import UNet
from dataset.sketch_dataset import SketchToImageDataset

# Configuration
torch.manual_seed(42)  # For reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
EPOCHS = 250

# Training transformations with strong augmentations
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(p=0.3),
    A.ColorJitter(p=0.4),
    A.GaussNoise(p=0.3),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, p=0.2),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
], additional_targets={'mask': 'image'})

# Validation transformations (minimal processing)
test_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
], additional_targets={'mask': 'image'})

# Dataset initialization
train_dataset = SketchToImageDataset(
    sketch_dir='./data/train/sketches',
    original_dir='./data/train/originals',
    transform=train_transform
)

test_dataset = SketchToImageDataset(
    sketch_dir='./data/test/sketches',
    original_dir='./data/test/originals',
    transform=test_transform
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model setup
model = UNet().to(device)
loss_fn = nn.L1Loss()  # Pixel-level reconstruction loss
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2)

# Start training
trainModel(EPOCHS, model, optimizer, scheduler, loss_fn, train_loader, test_loader, device)